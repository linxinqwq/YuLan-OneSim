from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from typing import Dict, Any, Optional, List
import os
import json
import asyncio
import time
import threading
import sys
import importlib.util
from loguru import logger
import uuid

from backend.models.simulation import (
    InitializeAgentsRequest, InitializeAgentsResponse, 
    GetAgentsRequest, GetAgentsResponse, AgentInfo,
    StartSimulationRequest, StartSimulationResponse,
    StopSimulationRequest, StopSimulationResponse,
    PauseSimulationRequest, PauseSimulationResponse,
    ResumeSimulationRequest, ResumeSimulationResponse,
    GetDecisionDataRequest, GetDecisionDataResponse,
    GetEventsResponse
)
from backend.utils.file_ops import create_directory, load_json
from backend.utils.model_management import load_model_if_needed
from backend.utils.websocket import connection_manager
from backend.routers.config import USER_CONFIGS,DEFAULT_CONFIG,MODEL_CONFIG_PATH


# 导入需要的OneSim组件
import onesim
from onesim.simulator import AgentFactory
from onesim.config import get_config,get_component_registry
from onesim.utils.work_graph import WorkGraph
from onesim.events.eventbus import get_event_bus,reset_event_bus
from onesim.monitor import MonitorManager
from onesim.simulator.sim_env import SimulationState

# 全局变量
# 统一的模拟环境注册表，包含所有环境信息、代理和状态
SIMULATION_REGISTRY = {}
# 聊天记录单独保存，避免数据结构复杂化
AGENT_CHAT_HISTORY = {}

router = APIRouter(
    tags=["simulation"],
    prefix="/simulation"
)


async def initialize_simulation(env_name: str, model_name: str = None) -> dict:
    """
    初始化模拟环境和相关组件。
    参考main.py和onesim.__init__.py的初始化流程，整合到config模块中。
    
    Args:
        env_name: 环境名称
        model_name: 可选的模型名称，如果未提供则从配置获取
        
    Returns:
        初始化状态的字典
    """
    # 检查环境是否存在
    scenes_root = os.path.join(os.getcwd(),"src", "envs")
    env_path = os.path.join(scenes_root, env_name)

    if not os.path.exists(env_path):
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在")

    try:
        # 获取配置 - 使用内存中的配置或默认配置
        if env_name in USER_CONFIGS:
            logger.info(f"使用内存中的环境配置: {env_name}")
            config_data = USER_CONFIGS[env_name]
        else:
            logger.info(f"使用默认环境配置: {env_name}")
            config_data = DEFAULT_CONFIG

        # 确保配置包含必要的环境信息
        config_data['env_name'] = env_name
        config_data['env_path'] = env_path

        # 如果未提供模型名称，尝试从配置获取
        if not model_name and "model" in config_data:
            if "chat" in config_data["model"]:
                model_name = config_data["model"]["chat"]
                logger.info(f"从配置获取模型: {model_name}")

        # 确保基础组件已初始化
        components_to_init = ["model"]

        # 添加其他可能需要的组件
        if config_data.get("monitor", {}).get("enabled", False):
            components_to_init.append("monitor")

        if config_data.get("database", {}).get("enabled", False):
            components_to_init.append("database")

        if config_data.get("distribution", {}).get("enabled", False):
            components_to_init.append("distribution")

        # 初始化必要的OneSim组件
        from onesim import init, COMPONENT_MODEL, COMPONENT_MONITOR, COMPONENT_DATABASE, COMPONENT_DISTRIBUTION

        component_registry = get_component_registry()
        component_registry.clear()
        # 转换组件名称为常量
        component_map = {
            "model": COMPONENT_MODEL,
            "monitor": COMPONENT_MONITOR,
            "database": COMPONENT_DATABASE,
            "distribution": COMPONENT_DISTRIBUTION
        }

        # 准备初始化配置
        init_components = [component_map[c] for c in components_to_init if c in component_map]

        # 初始化组件并获取配置
        config = await onesim.init(
            components=init_components,
            config_dict=config_data,  # 直接传入配置字典
            model_config_path=MODEL_CONFIG_PATH,
            #model_config_dict=model_config_dict
        )

        # 加载模型
        model = await load_model_if_needed(model_name)
        model_config_name = model.config_name
        # 加载SimEnv类定义
        import sys
        import importlib.util

        if scenes_root not in sys.path:
            sys.path.append(scenes_root)

        module_name = f"{env_name}.code.SimEnv"
        try:
            sim_env_module = importlib.import_module(module_name)
            if not hasattr(sim_env_module, "SimEnv"):
                raise AttributeError(f"模块 {module_name} 不包含名为 'SimEnv' 的类")

            SimEnv = getattr(sim_env_module, "SimEnv")
            logger.info(f"已加载环境类: {SimEnv.__name__}")
        except Exception as e:
            logger.error(f"加载环境类错误: {e}")
            raise Exception(f"无法加载环境类: {str(e)}")

        # 创建代理工厂
        agent_factory = AgentFactory(
            simulator_config=config.simulator_config,
            model_config_name=model_config_name,
            env_path=env_path,
            agent_config=config.agent_config
        )

        # 创建代理
        logger.info("创建代理实例")
        agents = await agent_factory.create_agents()

        # 构建工作流图
        logger.info("构建工作流图")
        actions_path = os.path.join(env_path, "actions.json")
        events_path = os.path.join(env_path, "events.json")

        # 解析操作和事件定义
        from onesim.config import parse_json
        actions = parse_json(actions_path)
        events = parse_json(events_path)

        # 创建工作流图并获取起始/结束节点
        work_graph = WorkGraph()
        work_graph.load_workflow_data(actions, events)
        start_agent_types = work_graph.get_start_agent_types()
        end_agent_types = work_graph.get_end_agent_types()

        start_agent_ids = agent_factory.get_agent_profile_ids(start_agent_types)
        end_agent_ids = agent_factory.get_agent_profile_ids(end_agent_types)

        # 添加环境关系
        for agent_type, ids in end_agent_ids.items():
            for agent_id in ids:
                agent_factory.add_env_relationship(agent_id)

        # 获取事件总线
        event_bus = get_event_bus()

        # 为分布式场景做检查
        is_distributed = False
        registry = get_component_registry()
        if registry.is_initialized(COMPONENT_DISTRIBUTION):
            node = registry.get_instance(COMPONENT_DISTRIBUTION)
            is_distributed = True
            logger.info(f"检测到分布式模式: {node.role}")

        # 初始化数据跟踪ID（如果启用数据库）
        trail_id = None
        if registry.is_initialized(COMPONENT_DATABASE):
            try:
                from onesim.data import ScenarioManager, TrailManager
                import time

                # 创建或获取情景ID
                scenario_mgr = ScenarioManager()
                env_config = config.simulator_config.environment

                # 尝试找到现有场景
                scenarios = await scenario_mgr.get_scenario_by_name(name=env_name, exact_match=True)
                scenario_id = None

                if scenarios and len(scenarios) > 0:
                    for scenario in scenarios:
                        if scenario['name'] == env_name:
                            scenario_id = scenario['scenario_id']
                            logger.info(f"使用现有场景ID {scenario_id} for {env_name}")
                            break

                if scenario_id is None:
                    # 创建新场景
                    env_path=os.path.join("src","envs",env_name)
                    scenario_id = await scenario_mgr.create_scenario(
                        name=env_name,
                        folder_path=env_path,
                        description=env_config.get('description', f"Simulation scenario for {env_name}"),
                        tags={
                            "domain": env_config.get('domain', ''), 
                            "version": env_config.get('version', '1.0')
                        }
                    )
                    logger.info(f"创建新场景ID {scenario_id} for {env_name}")

                # 创建trail
                trail_mgr = TrailManager()
                trail_name = f"{env_name}_run_{time.strftime('%Y%m%d_%H%M%S')}"
                trail_id = await trail_mgr.create_trail(
                    scenario_id=scenario_id,
                    name=trail_name,
                    description=f"Simulation run for {env_name}",
                    config=config.simulator_config.to_dict()
                )
                logger.info(f"创建数据跟踪ID {trail_id} 用于数据存储")
            except Exception as e:
                logger.error(f"初始化数据存储错误: {e}, 继续而不存储数据")

        # 创建环境实例
        simulator_config = config.simulator_config
        env_settings = simulator_config.environment

        # 创建带时间戳的输出目录
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        runs_dir = os.path.join(env_path, "runs")
        os.makedirs(runs_dir, exist_ok=True)
        output_dir = os.path.join(runs_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        sim_env = SimEnv(
            env_name,
            event_bus,
            {},  # initial data
            start_agent_ids,
            end_agent_ids,
            env_settings,
            agents,
            env_path,
            trail_id,  # Pass trail_id to environment
            None,  # intervention_engine
            output_dir,  # Pass the timestamped output directory
        )
        end_events = work_graph.get_end_events()
        # Register termination events
        for event_name in end_events:
            sim_env.register_event(event_name, 'terminate')

        simulation_id = str(uuid.uuid4())

        # 在全局注册表中存储代理工厂和代理 - 与simulation.py中的格式保持一致
        SIMULATION_REGISTRY[env_name] = {
            "agent_factory": agent_factory,
            "agents": agents,
            "initialized": True,
            "running": False,
            "config": config_data,
            "event_bus": event_bus,
            "work_graph": work_graph,
            "start_agent_ids": start_agent_ids,
            "end_agent_ids": end_agent_ids,
            "SimEnv": SimEnv,
            "env_path": env_path,
            "end_events": work_graph.get_end_events(),
            "simulation_id": simulation_id,
            "trail_id": trail_id,
            "sim_env": sim_env,  # Store the simulation environment
            # 添加状态信息
            "status": "initialized",
            "metrics": {},
            "step": 0,
            "start_time": None,
            "pause_time": None,
            "events": []
        }

        created_agents=[]
        for agent_type in agents:
            for agent_id, agent in agents[agent_type].items():
                created_agents.append(AgentInfo(
                    id=agent_id,
                    type=agent_type,
                    profile=agent.get_profile(include_private=True)
                ))
        # 返回初始化状态
        result = {
            "env_name": env_name,
            "config_applied": True,
            "agents": created_agents,
            "agent_count": sum(len(agents[agent_type]) for agent_type in agents),
            "is_distributed": is_distributed,
            "trail_id": trail_id,
            "components_initialized": {
                component: registry.is_initialized(component_map[component])
                for component in components_to_init if component in component_map
            },
            "workflow": {
                "start_agent_types": start_agent_types,
                "end_agent_types": end_agent_types,
                "start_agent_ids": start_agent_ids,
                "end_agent_ids": end_agent_ids,
                "end_events": work_graph.get_end_events()
            },
            "ready_for_simulation": True
        }

        logger.info(f"环境 '{env_name}' 初始化成功")
        return result

    except Exception as e:
        logger.error(f"初始化模拟环境出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化模拟环境失败: {str(e)}")


@router.post("/initialize")
async def initialize_simulation_endpoint(env_name: str, model_name: Optional[str] = None):
    """初始化模拟环境和相关组件的端点"""
    try:
        result = await initialize_simulation(env_name, model_name)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"初始化模拟出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化模拟失败: {str(e)}")


@router.post("/get_agents", response_model=GetAgentsResponse)
def get_agents_info(data: GetAgentsRequest):
    """获取代理信息"""
    env_name = data.env_name
    agent_type = data.agent_type
    
    # 检查环境是否存在
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在或未初始化")
    
    agents = []
    
    # 获取环境中的代理
    env_agents = SIMULATION_REGISTRY[env_name]["agents"]
    
    # 过滤代理类型（如果指定）
    for agent_type_key, agents_of_type in env_agents.items():
        if agent_type is None or agent_type_key == agent_type:
            for agent_id, agent in agents_of_type.items():
                agents.append(AgentInfo(
                    id=agent_id,
                    type=agent_type_key,
                    profile=agent.get_profile(include_private=True)
                ))
    
    return GetAgentsResponse(
        agents=agents,
        count=len(agents)
    )

@router.post("/start", response_model=StartSimulationResponse)
async def start_simulation(data: StartSimulationRequest):
    """启动仿真"""
    env_name = data.env_name

    # Ensure only one simulation runs at a time
    env_names_to_remove = []
    for other_env_name, other_registry in SIMULATION_REGISTRY.items():
        if other_env_name != env_name and other_registry.get("running", False):
            logger.info(f"Stopping existing running simulation: {other_env_name} to start {env_name}")
            try:
                await stop_simulation(StopSimulationRequest(env_name=other_env_name))
                env_names_to_remove.append(other_env_name)
            except Exception as stop_err:
                logger.error(f"Error stopping simulation {other_env_name}: {stop_err}")
                # Decide if we should still try to remove or halt starting the new one
                # For now, we log the error and mark for removal if stop seemed to proceed partially
                if not SIMULATION_REGISTRY.get(other_env_name, {}).get("running", False):
                    env_names_to_remove.append(other_env_name)

    for name_to_remove in env_names_to_remove:
        if name_to_remove in SIMULATION_REGISTRY:
            logger.info(
                f"Removing registry entry for stopped simulation: {name_to_remove}"
            )
            try:
                del SIMULATION_REGISTRY[name_to_remove]
            except KeyError:
                logger.warning(f"Registry entry for {name_to_remove} already removed.")

    # Check if the requested environment exists
    if env_name not in SIMULATION_REGISTRY:
        # 如果环境不存在，则直接报错
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在或初始化失败")

    registry = SIMULATION_REGISTRY[env_name]

    # 检查环境是否需要重新初始化（停止后再启动）
    if registry.get("status") == "stopped" or registry.get("needs_reinit", False):
        logger.info(f"检测到环境 '{env_name}' 需要重新初始化")
        try:
            # 重置事件总线
            reset_event_bus()
            logger.info(f"已重置全局事件总线")
            AGENT_CHAT_HISTORY.clear()
            # 获取模型名称
            model_name = None
            if "config" in registry and "model" in registry["config"]:
                if "chat" in registry["config"]["model"]:
                    model_name = registry["config"]["model"]["chat"]

            # 重新初始化环境
            await initialize_simulation(env_name, model_name)
            logger.info(f"环境 '{env_name}' 自动重初始化成功")

            # 获取重新初始化后的注册表
            registry = SIMULATION_REGISTRY[env_name]

        except Exception as e:
            logger.error(f"自动重初始化环境 '{env_name}' 失败: {str(e)}")
            raise HTTPException(status_code=500, 
                detail=f"环境 '{env_name}' 需要重新初始化，但自动重初始化失败: {str(e)}")

    # 检查初始化状态
    if not registry.get("initialized", False):
        raise HTTPException(status_code=400, detail=f"环境 '{env_name}' 未初始化。请先保存配置。")

    # 检查是否已经在运行
    if registry.get("running", False):
        return StartSimulationResponse(
            success=False,
            message=f"环境 '{env_name}' 仿真已经在运行中",
            simulation_id=registry.get("simulation_id", f"sim_{env_name}")
        )

    try:
        # 创建模拟ID
        simulation_id = str(uuid.uuid4())

        # 获取初始化数据
        agents = registry["agents"]
        event_bus = registry["event_bus"]
        start_agent_ids = registry["start_agent_ids"]
        end_agent_ids = registry["end_agent_ids"]
        sim_env = registry["sim_env"]

        if hasattr(sim_env, "simulation_id"):
            sim_env.simulation_id = simulation_id

        # 确保模拟环境状态正确
        if hasattr(sim_env, "_state") and sim_env._state == SimulationState.TERMINATED:
            logger.info(f"重置模拟环境状态从TERMINATED到INITIALIZED，环境: '{env_name}'")
            await sim_env.set_simulation_state(SimulationState.INITIALIZED, reason="restart")

        # 重置代理状态
        for agent_type in agents:
            for agent_id, agent in agents[agent_type].items():
                if hasattr(agent, "stopped") and agent.stopped:
                    agent.stopped = False
                    logger.debug(f"重置代理停止状态: {agent_id}")

        # 注册环境到事件总线
        event_bus.register_agent("ENV", sim_env)

        # 注册代理到事件总线并设置环境
        for agent_type in agents:
            for agent_id, agent in agents[agent_type].items():
                if hasattr(agent, "set_env"):
                    agent.set_env(sim_env)
                event_bus.register_agent(agent_id, agent)

        logger.info(
            f"已注册环境和 {sum(len(agents[t]) for t in agents)} 个代理到事件总线"
        )

        simulator_registry = get_component_registry()
        if simulator_registry.is_initialized("monitor"):
            await MonitorManager.setup_metrics(
                env=sim_env
            )

        # 更新注册表信息
        registry["simulation_id"] = simulation_id
        registry["running"] = True
        registry["paused"] = False
        registry["needs_reinit"] = False  # 清除重初始化标记

        # 启动模拟任务
        async def run_simulation_tasks():
            try:
                # 创建终止事件
                termination_event = asyncio.Event()
                registry["termination_event"] = termination_event

                # 获取环境任务
                env_tasks = await sim_env.run()

                # 创建代理任务
                agent_tasks = []
                for agent_type in agents:
                    for agent_id, agent in agents[agent_type].items():
                        if hasattr(agent, "run"):
                            agent_tasks.append(asyncio.create_task(agent.run()))

                # 运行事件总线任务
                event_bus_task = asyncio.create_task(event_bus.run())

                # 全部任务列表
                all_tasks = [event_bus_task] + agent_tasks + env_tasks

                # 存储任务列表
                registry["tasks"] = all_tasks

                # 等待任务完成或终止信号
                while not termination_event.is_set():
                    try:
                        # 检查任务状态
                        done, pending = await asyncio.wait(
                            [asyncio.create_task(termination_event.wait())] + all_tasks,
                            timeout=1.0,
                            return_when=asyncio.FIRST_COMPLETED
                        )

                        # 检查终止信号
                        if termination_event.is_set():
                            logger.info(f"收到终止信号，停止环境 '{env_name}' 模拟")
                            await sim_env.stop_simulation()
                            break

                        # 检查任务状态
                        for task in done:
                            if task.done():
                                if task.exception():
                                    logger.error(f"任务执行异常: {task.exception()}")
                                    await sim_env.stop_simulation()
                                    termination_event.set()
                                    break

                        # 如果事件总线停止运行，结束模拟
                        if hasattr(event_bus, "_running") and not event_bus._running:
                            logger.warning(f"事件总线已停止运行，环境: '{env_name}'")
                            termination_event.set()
                            break

                        # 继续等待
                        if not termination_event.is_set():
                            continue

                    except asyncio.CancelledError:
                        logger.info(f"模拟任务被取消，环境: '{env_name}'")
                        break
                    except Exception as e:
                        logger.error(f"模拟执行错误: {e}")
                        await sim_env.stop_simulation()
                        termination_event.set()

                # 清理资源
                logger.info(f"模拟已结束，清理资源，环境: '{env_name}'")
                registry["running"] = False

                # 关闭环境的WebSocket连接
                try:
                    await connection_manager.close_websocket_by_env_name(env_name)
                    logger.info(f"已关闭环境 '{env_name}' 的WebSocket连接")
                except Exception as e:
                    logger.error(f"关闭WebSocket连接时出错: {e}")

                # 取消所有任务
                for task in all_tasks:
                    if not task.done():
                        task.cancel()

                # 等待任务取消完成
                await asyncio.gather(*all_tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"模拟执行错误: {e}")
                registry["running"] = False

                # 关闭环境的WebSocket连接
                try:
                    await connection_manager.close_websocket_by_env_name(env_name)
                    logger.info(f"已关闭环境 '{env_name}' 的WebSocket连接")
                except Exception as e:
                    logger.error(f"关闭WebSocket连接时出错: {e}")

                # 取消任务
                if 'all_tasks' in locals():
                    for task in all_tasks:
                        if not task.done():
                            task.cancel()

                    try:
                        await asyncio.gather(*all_tasks, return_exceptions=True)
                    except Exception:
                        pass

        # 后台启动模拟任务
        asyncio.create_task(run_simulation_tasks())

        # 准备统计信息
        agent_count = sum(len(agents[agent_type]) for agent_type in agents)

        # 更新模拟状态
        registry["status"] = "running"
        registry["start_time"] = time.time()
        registry["pause_time"] = None

        return StartSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真已启动，共 {agent_count} 个代理",
            simulation_id=simulation_id
        )

    except Exception as e:
        logger.error(f"启动仿真错误: {e}")
        raise HTTPException(status_code=500, detail=f"启动仿真失败: {str(e)}")

@router.post("/stop", response_model=StopSimulationResponse)
async def stop_simulation(data: StopSimulationRequest):
    """停止仿真"""
    env_name = data.env_name

    # Check if simulation exists
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"未找到环境 '{env_name}' 的仿真")

    registry = SIMULATION_REGISTRY[env_name]
    if not registry.get("running", False):
        return StopSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真未在运行"
        )

    try:
        # Get the simulation environment
        sim_env = registry.get("sim_env")
        event_bus = registry.get("event_bus")
        agents = registry.get("agents")

        # Retrieve the MonitorManager if available
        monitor_manager: Optional[MonitorManager] = None
        try:
            monitor_registry = get_component_registry()
            if monitor_registry.is_initialized("monitor"):
                monitor_manager = monitor_registry.get_instance("monitor")
        except Exception as e:
            logger.warning(f"获取 MonitorManager 实例时出错: {e}")

        # If simulation is paused, resume it first to allow clean shutdown
        if registry.get("paused", False):
            logger.info(f"在停止前恢复暂停的仿真，环境: '{env_name}'")
            # Set the pause event to signal tasks to continue
            if "pause_event" in registry and isinstance(registry["pause_event"], asyncio.Event):
                registry["pause_event"].set()
            registry["paused"] = False

        # Set termination event if it exists
        if "termination_event" in registry and isinstance(registry["termination_event"], asyncio.Event):
            logger.info(f"设置终止事件，环境: '{env_name}'")
            registry["termination_event"].set()

        # Also call stop_simulation on the sim_env
        if sim_env and hasattr(sim_env, "stop_simulation"):
            logger.info(f"停止仿真环境: '{env_name}'")
            await sim_env.stop_simulation()

        # Stop monitor metrics if MonitorManager is available
        if monitor_manager and hasattr(monitor_manager, "stop_all_metrics"):
            logger.info(f"停止环境 '{env_name}' 的监控指标")
            await monitor_manager.stop_all_metrics()

        # Cancel all tasks if they exist
        if "tasks" in registry and registry["tasks"]:
            logger.info(f"取消环境 '{env_name}' 的 {len(registry['tasks'])} 个任务")
            for task in registry["tasks"]:
                if not task.done():
                    task.cancel()

            # Wait for tasks to be properly canceled
            try:
                await asyncio.gather(*registry["tasks"], return_exceptions=True)
            except Exception as e:
                logger.error(f"等待任务取消时出错: {e}")

        # 清除事件总线上的代理注册
        if event_bus:
            reset_event_bus()

        # component_registry = get_component_registry()
        # component_registry.clear()
        # 清理资源引用
        logger.info(f"清理环境 '{env_name}' 的资源引用")
        if "termination_event" in registry:
            del registry["termination_event"]  # 删除终止事件引用

        if "tasks" in registry:
            del registry["tasks"]  # 清理任务引用

        # Mark simulation as not running
        registry["running"] = False
        registry["paused"] = False
        registry["status"] = "stopped"
        registry["needs_reinit"] = True  # 标记为需要重新初始化

        # 广播停止事件
        stop_event = {
            "type": "EndEvent",
            "step": registry["step"],
            "time": time.time(),
            "reason": "user_requested"
        }

        await connection_manager.broadcast_event(env_name, stop_event)

        await connection_manager.close_websocket_by_env_name(env_name)
        return StopSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真已停止"
        )
    except Exception as e:
        logger.error(f"停止仿真错误: {e}")
        raise HTTPException(status_code=500, detail=f"停止仿真失败: {str(e)}")

@router.post("/pause", response_model=PauseSimulationResponse)
async def pause_simulation(data: PauseSimulationRequest):
    """暂停仿真"""
    env_name = data.env_name
    
    # Check if simulation exists
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"未找到环境 '{env_name}' 的仿真")
    
    registry = SIMULATION_REGISTRY[env_name]
    if not registry.get("running", False):
        return PauseSimulationResponse(
            success=False,
            message=f"环境 '{env_name}' 仿真未在运行",
            is_paused=False
        )
    
    # Check if the simulation is already paused
    if registry.get("paused", False):
        return PauseSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真已经处于暂停状态",
            is_paused=True
        )
    
    try:
        # Get the simulation environment
        sim_env = registry.get("sim_env")
        
        # 优先使用SimEnv的暂停方法，它会内部处理EventBus的暂停
        if sim_env and hasattr(sim_env, "pause_simulation"):
            logger.info(f"暂停环境 '{env_name}' 的仿真环境")
            await sim_env.pause_simulation()
        else:
            # 仅在SimEnv不可用时才直接暂停EventBus
            event_bus = registry.get("event_bus")
            if event_bus:
                if hasattr(event_bus, "pause"):
                    logger.info(f"直接暂停环境 '{env_name}' 的事件总线")
                    await event_bus.pause()
                else:
                    # If pause method doesn't exist, we'll set a flag for custom handling
                    if not hasattr(event_bus, "_paused"):
                        event_bus._paused = False
                    event_bus._paused = True
                    logger.info(f"在环境 '{env_name}' 的事件总线上设置暂停标志")
        
        # Set the pause flag in registry
        registry["paused"] = True
        registry["status"] = "paused"
        registry["pause_time"] = time.time()
        
        # 创建暂停事件
        pause_event = {
            "type": "PauseEvent",
            "step": registry["step"],
            "time": time.time()
        }
        
        # 广播暂停事件
        await connection_manager.broadcast_event(env_name, pause_event)
        
        return PauseSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真已暂停",
            is_paused=True
        )
    except Exception as e:
        logger.error(f"暂停仿真错误: {e}")
        raise HTTPException(status_code=500, detail=f"暂停仿真失败: {str(e)}")

@router.post("/resume", response_model=ResumeSimulationResponse)
async def resume_simulation(data: ResumeSimulationRequest):
    """恢复仿真"""
    env_name = data.env_name
    
    # Check if simulation exists
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"未找到环境 '{env_name}' 的仿真")
    
    registry = SIMULATION_REGISTRY[env_name]
    
    # Check if the simulation is running but not paused
    if registry.get("running", False) and not registry.get("paused", False):
        return ResumeSimulationResponse(
            success=False,
            message=f"环境 '{env_name}' 仿真已经在运行中",
            is_running=True
        )
    
    # Check if the simulation is not running at all
    if not registry.get("running", False):
        return ResumeSimulationResponse(
            success=False,
            message=f"环境 '{env_name}' 仿真未在运行",
            is_running=False
        )
    
    try:
        # Get the simulation environment
        sim_env = registry.get("sim_env")
        
        # 优先使用SimEnv的恢复方法，它会内部处理EventBus的恢复
        if sim_env and hasattr(sim_env, "resume_simulation"):
            logger.info(f"恢复环境 '{env_name}' 的仿真环境")
            await sim_env.resume_simulation()
        else:
            # 仅在SimEnv不可用时才直接恢复EventBus
            event_bus = registry.get("event_bus")
            if event_bus:
                if hasattr(event_bus, "resume"):
                    logger.info(f"直接恢复环境 '{env_name}' 的事件总线")
                    await event_bus.resume()
                elif hasattr(event_bus, "_paused"):
                    # Clear the custom pause flag
                    event_bus._paused = False
                    logger.info(f"清除环境 '{env_name}' 的事件总线上的暂停标志")
        
        # Clear the pause flag
        registry["paused"] = False
        registry["status"] = "running"
        
        # 创建恢复事件
        resume_event = {
            "type": "ResumeEvent",
            "step": registry["step"],
            "time": time.time()
        }
        
        # 广播恢复事件
        await connection_manager.broadcast_event(env_name, resume_event)
        
        return ResumeSimulationResponse(
            success=True,
            message=f"环境 '{env_name}' 仿真已恢复",
            is_running=True
        )
    except Exception as e:
        logger.error(f"恢复仿真错误: {e}")
        raise HTTPException(status_code=500, detail=f"恢复仿真失败: {str(e)}")

@router.get("/{env_name}/available_metrics")
async def get_available_metrics(env_name: str):
    """获取可用指标"""
    # 检查环境是否存在
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在或未初始化")
    
    # 返回可用指标
    return {
        "agent_metrics": [
            "activity_level",
            "social_connections",
            "resource_usage"
        ],
        "env_metrics": [
            "population",
            "resource_level",
            "social_stability"
        ]
    }

@router.get("/{env_name}/events", response_model=GetEventsResponse)
async def get_simulation_events(env_name: str):
    """获取仿真事件"""
    # 检查环境是否存在
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在或未初始化")

    # 获取模拟环境
    registry = SIMULATION_REGISTRY[env_name]
    sim_env = registry.get("sim_env")

    events = []
    if sim_env and hasattr(sim_env, "_pending_events"):
        # 从模拟环境中获取事件，而不是使用控制事件（暂停、停止等）
        events = sim_env._pending_events

    return GetEventsResponse(
        success=True,
        message=f"获取事件成功: {env_name}",
        events=events
    )


@router.get("/{env_name}/token_usage")
async def get_token_usage_stats(env_name: str):
    """获取当前统计的token使用情况，以及对应的模型的信息"""
    # 检查环境是否存在
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(
            status_code=404, detail=f"环境 '{env_name}' 不存在或未初始化"
        )

    try:
        # 导入token使用统计功能
        from onesim.models.utils.token_usage import get_token_usage_stats
        from onesim.models import ModelManager

        # 获取token使用统计
        token_stats = get_token_usage_stats()

        # 获取模型管理器实例以获取模型配置信息
        model_manager = ModelManager.get_instance()

        # 构建模型信息列表
        model_info_list = []
        model_usage = token_stats.get("model_usage", {})

        for model_name, usage_data in model_usage.items():
            # 查找对应的模型配置
            model_config = None
            config_name = None

            # 遍历所有模型配置，找到匹配的模型
            for config_key, config in model_manager.model_configs.items():
                # 检查模型名称是否匹配
                if (
                    config.get("model_name") == model_name
                    or config.get("config_name") == model_name
                    or model_name in config.get("model_name", "")
                ):
                    model_config = config
                    config_name = config_key
                    break

            # 构建模型信息
            model_info = {
                "model_name": model_name,
                "model_config_name": config_name,
                "provider": (
                    model_config.get("provider", "unknown")
                    if model_config
                    else "unknown"
                ),
                "category": (
                    model_config.get("category", "unknown")
                    if model_config
                    else "unknown"
                ),
                "token_usage": {
                    "total_tokens": usage_data.get("total_tokens", 0),
                    "total_prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "total_completion_tokens": usage_data.get("completion_tokens", 0),
                    "request_count": usage_data.get("request_count", 0),
                },
            }

            # 如果有模型配置，添加更多详细信息
            if model_config:
                # 添加client_args
                if "client_args" in model_config:
                    model_info["client_args"] = model_config["client_args"]

                # 添加生成参数(如果有)
                if "generate_args" in model_config:
                    model_info["generate_args"] = model_config["generate_args"]

            model_info_list.append(model_info)

        # 构建响应
        response = {
            "success": True,
            "env_name": env_name,
            "total_statistics": {
                "total_tokens": token_stats.get("total_tokens", 0),
                "total_prompt_tokens": token_stats.get("total_prompt_tokens", 0),
                "total_completion_tokens": token_stats.get(
                    "total_completion_tokens", 0
                ),
                "request_count": token_stats.get("request_count", 0),
                "elapsed_time_seconds": token_stats.get("elapsed_time_seconds", 0),
                "tokens_per_second": token_stats.get("tokens_per_second", 0),
            },
            "models": model_info_list,
        }

        return response

    except ImportError:
        logger.warning("Token使用模块不可用")
        raise HTTPException(status_code=503, detail="Token使用统计功能不可用")
    except Exception as e:
        logger.error(f"获取token使用统计时出错: {e}")
        raise HTTPException(status_code=500, detail=f"获取token使用统计失败: {str(e)}")


@router.get("/list_environments")
async def list_environments():
    """获取当前所有已初始化的环境名称列表"""
    return {
        "success": True,
        "environments": list(SIMULATION_REGISTRY.keys())
    }

@router.get("/registry/{env_name}")
async def get_simulation_registry(env_name: str = ""):
    """
    获取模拟环境注册表信息
    
    Args:
        env_name: 环境名称，如果为空则返回整个注册表
        
    Returns:
        注册表信息（整个或特定环境）
    """
    try:
        # 处理空路径参数
        if env_name == "":
            # 返回整个注册表的安全副本（排除敏感数据）
            safe_registry = {}
            for env, registry in SIMULATION_REGISTRY.items():
                # 创建不含敏感数据的副本
                safe_registry[env] = {
                    "initialized": registry.get("initialized", False),
                    "running": registry.get("running", False),
                    "paused": registry.get("paused", False),
                    "status": registry.get("status", "unknown"),
                    "step": registry.get("step", 0),
                    "start_time": registry.get("start_time"),
                    "pause_time": registry.get("pause_time"),
                    "simulation_id": registry.get("simulation_id"),
                    "agent_count": sum(len(registry.get("agents", {}).get(agent_type, {})) 
                                    for agent_type in registry.get("agents", {})),
                    "env_path": registry.get("env_path")
                }
            return {
                "success": True,
                "message": "获取全部环境注册信息成功",
                "registry": safe_registry
            }
        
        # 检查环境是否存在
        if env_name not in SIMULATION_REGISTRY:
            return {
                "success": False,
                "message": f"环境 '{env_name}' 不存在或未初始化",
                "registry": None
            }
        
        # 获取特定环境的注册表
        registry = SIMULATION_REGISTRY[env_name]
        
        # 创建安全副本（排除敏感数据和大型对象引用）
        safe_registry = {
            "initialized": registry.get("initialized", False),
            "running": registry.get("running", False),
            "paused": registry.get("paused", False),
            "status": registry.get("status", "unknown"),
            "step": registry.get("step", 0),
            "start_time": registry.get("start_time"),
            "pause_time": registry.get("pause_time"),
            "simulation_id": registry.get("simulation_id"),
            "trail_id": registry.get("trail_id"),
            "env_path": registry.get("env_path"),
            "end_events": registry.get("end_events", []),
            "needs_reinit": registry.get("needs_reinit", False),
            "agent_types": list(registry.get("agents", {}).keys()),
            "agent_counts": {
                agent_type: len(agents) 
                for agent_type, agents in registry.get("agents", {}).items()
            },
            "config": registry.get("config", {})
        }
        
        # 添加代理列表
        safe_registry["agent_list"] = []
        for agent_type, agents in registry.get("agents", {}).items():
            for agent_id in agents:
                safe_registry["agent_list"].append({
                    "id": agent_id,
                    "type": agent_type
                })
        
        return {
            "success": True,
            "message": f"获取环境 '{env_name}' 注册信息成功",
            "registry": safe_registry
        }
        
    except Exception as e:
        logger.error(f"获取注册表信息错误: {e}")
        return {
            "success": False,
            "message": f"获取注册表信息失败: {str(e)}",
            "registry": None
        }


@router.websocket("/ws/{env_name}")
async def websocket_endpoint(websocket: WebSocket, env_name: str):
    """WebSocket连接端点，包含超时处理"""
    # 连接超时设置（秒）
    DISCONNECT_TIMEOUT = 120  # 2分钟断开连接后结束模拟
    EVENT_TIMEOUT = 120  # 2分钟没有新事件则结束模拟

    await connection_manager.connect(websocket, env_name)
    logger.info(f"WebSocket connected: {env_name}")
    last_event_time = time.time()

    try:
        # 发送初始状态
        if env_name in SIMULATION_REGISTRY:
            registry = SIMULATION_REGISTRY[env_name]
            await websocket.send_json({
                "type": "simulation_state",
                "env_name": env_name,
                "status": registry["status"],
                "step": registry["step"],
                "time": time.time()
            })

        # 等待消息的同时检查超时
        while True:
            try:
                # 使用wait_for设置接收消息的超时时间
                data = await asyncio.wait_for(
                    websocket.receive_text(), 
                    timeout=30  # 每30秒检查一次超时状态
                )

                # 在这里可以处理接收到的WebSocket消息的逻辑

            except asyncio.TimeoutError:
                # 检查是否超出没有事件的时间限制
                current_time = time.time()

                # 检查模拟是否正在运行
                if env_name in SIMULATION_REGISTRY:
                    # 如果是暂停状态，不触发超时
                    if SIMULATION_REGISTRY[env_name].get("paused", False) or SIMULATION_REGISTRY[env_name].get("status") == "paused":
                        # 暂停状态下重置最后事件时间，避免暂停后恢复立即触发超时
                        last_event_time = current_time
                        continue

                    # 获取模拟环境的最后事件时间
                    sim_env = SIMULATION_REGISTRY[env_name].get("sim_env")
                    should_stop = False

                    if sim_env and hasattr(sim_env, '_last_event_time'):
                        sim_last_event_time = sim_env._last_event_time
                        # 只有在运行状态且超时时才触发结束模拟
                        should_stop = (
                            SIMULATION_REGISTRY[env_name].get("running", False)
                            and not SIMULATION_REGISTRY[env_name].get("paused", False)
                            and current_time - sim_last_event_time > EVENT_TIMEOUT
                        )
                    else:
                        # 如果无法获取sim_env的时间，回退到原来的逻辑
                        should_stop = (
                            SIMULATION_REGISTRY[env_name].get("running", False)
                            and not SIMULATION_REGISTRY[env_name].get("paused", False)
                            and current_time - last_event_time > EVENT_TIMEOUT
                        )

                    if should_stop:
                        logger.warning(f"环境 '{env_name}' 超过 {EVENT_TIMEOUT} 秒没有新事件，自动结束模拟")

                        try:
                            # 调用停止模拟的逻辑
                            from backend.models.simulation import StopSimulationRequest
                            stop_data = StopSimulationRequest(env_name=env_name)
                            await stop_simulation(stop_data)
                            SIMULATION_REGISTRY[env_name]["running"] = False

                        except Exception as e:
                            logger.error(f"停止模拟时发生错误: {e}")
                        break  # 结束WebSocket循环

                # 如果没超时，就继续循环
                continue

    except WebSocketDisconnect:
        logger.info(f"WebSocket客户端断开连接: {env_name}")

        # 检查是否有其他连接
        has_other_connections = False
        if env_name in connection_manager.active_connections:
            # 移除当前连接
            if websocket in connection_manager.active_connections[env_name]:
                connection_manager.active_connections[env_name].remove(websocket)

            # 检查是否还有其他连接
            has_other_connections = len(connection_manager.active_connections[env_name]) > 0

        # 如果没有其他连接，并且模拟正在运行，启动定时任务检查断开超时
        if not has_other_connections and env_name in SIMULATION_REGISTRY and SIMULATION_REGISTRY[env_name].get("running", False):
            try:
                # 创建一个等待任务
                async def disconnect_timeout_handler():
                    logger.info(f"等待 {DISCONNECT_TIMEOUT} 秒后检查环境 '{env_name}' 的连接状态")
                    await asyncio.sleep(DISCONNECT_TIMEOUT)

                    # 再次检查是否仍然没有连接
                    if (env_name not in connection_manager.active_connections or 
                        len(connection_manager.active_connections[env_name]) == 0):
                        logger.warning(f"环境 '{env_name}' 的所有客户端已断开连接 {DISCONNECT_TIMEOUT} 秒，自动结束模拟")

                        # 调用停止模拟的逻辑
                        if env_name in SIMULATION_REGISTRY and SIMULATION_REGISTRY[env_name].get("running", False):
                            from backend.models.simulation import StopSimulationRequest
                            stop_data = StopSimulationRequest(env_name=env_name)
                            await stop_simulation(stop_data)
                            SIMULATION_REGISTRY[env_name]["running"] = False
                # 启动超时任务
                asyncio.create_task(disconnect_timeout_handler())
            except Exception as e:
                logger.error(f"设置断开连接超时处理时出错: {e}")
    except Exception as e:
        logger.error(f"WebSocket处理时出错: {e}")
    finally:
        # 确保连接已断开
        connection_manager.disconnect(websocket, env_name)

# 添加广播记忆接口
from pydantic import BaseModel

class BroadcastMessageRequest(BaseModel):
    env_name: str
    message: str

@router.post("/broadcast")
async def broadcast_memory(data: BroadcastMessageRequest):
    """
    广播记忆给所有代理
    
    Args:
        data: 包含环境名称和记忆内容的请求体
    
    Returns:
        广播结果
    """
    env_name = data.env_name
    memory = data.message
    
    # 检查环境是否存在
    if env_name not in SIMULATION_REGISTRY:
        raise HTTPException(status_code=404, detail=f"环境 '{env_name}' 不存在或未初始化")
    
    # # 检查环境是否正在运行
    registry = SIMULATION_REGISTRY[env_name]
    # if not registry.get("running", False):
    #     raise HTTPException(status_code=400, detail=f"环境 '{env_name}' 仿真未在运行")
    
    try:
        # 获取所有代理
        agents = registry["agents"]
        broadcast_count = 0
        
        # 遍历所有代理，调用add_memory方法
        for agent_type in agents:
            for agent_id, agent in agents[agent_type].items():
                if hasattr(agent, "add_memory") and callable(agent.add_memory):
                    # 调用代理的add_memory方法添加记忆
                    await agent.add_memory(memory)
                    broadcast_count += 1
        
        # 记录广播事件
        broadcast_event = {
            "type": "BroadcastMessageEvent",
            "step": registry["step"],
            "time": int(time.time()),
            "message": memory,
            "target_count": broadcast_count
        }
        
        # 广播事件
        await connection_manager.broadcast_event(env_name, broadcast_event)
        
        return {
            "success": True, 
            "message": f"成功向 {broadcast_count} 个代理广播记忆",
            "broadcast_count": broadcast_count
        }
    except Exception as e:
        logger.error(f"广播记忆错误: {e}")
        raise HTTPException(status_code=500, detail=f"广播记忆失败: {str(e)}")
