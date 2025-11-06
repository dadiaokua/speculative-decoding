"""GPU功率和性能监控模块

本模块用于在基准测试运行期间监控GPU的功率消耗、利用率、内存使用和温度。

主要功能：
1. 实时采集GPU硬件指标（功率、温度、利用率等）
2. 通过时间积分法计算能耗（梯形积分）
3. 记录性能指标（吞吐量、延迟、token生成数等）
4. 计算能效指标（每焦耳生成的token数）

使用方法：
    monitor = GPUMonitor(gpu_ids=[0,1,2], sampling_interval=0.5)
    monitor.start()
    # ... 运行benchmark ...
    monitor.stop()
    results = monitor.get_results()
    print_gpu_summary(results)
"""

import subprocess
import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict
import os
from termcolor import colored


@dataclass
class GPUSnapshot:
    """GPU指标的单次快照
    
    存储某一时刻所有GPU硬件指标和性能指标的快照数据。
    每个采样间隔（如0.5秒）会生成一次快照。
    """
    # 时间戳和标识
    timestamp: float            # 采样时间戳（Unix时间）
    gpu_id: int                 # GPU编号（0-7）
    
    # 功率指标（通过nvidia-smi查询）
    power_draw: float           # 当前功率消耗（瓦特）
    power_limit: float          # 功率上限（瓦特）
    
    # 利用率指标
    utilization_gpu: float      # GPU计算利用率（百分比 0-100）
    utilization_memory: float   # 显存带宽利用率（百分比 0-100）
    
    # 显存使用
    memory_used: int            # 已使用显存（MB）
    memory_total: int           # 总显存（MB）
    
    # 温度和频率
    temperature: float          # GPU温度（摄氏度）
    clock_graphics: int         # GPU核心频率（MHz）
    clock_memory: int           # 显存频率（MHz）
    
    # 性能指标（由benchmark回调函数设置，可选）
    total_tokens_generated: int = 0  # 截至当前已生成的总token数
    total_tokens_accepted: int = 0   # 截至当前已接受的token数（推测解码）
    requests_completed: int = 0      # 截至当前已完成的请求数
    throughput: float = 0.0          # 当前吞吐量（tokens/秒）
    avg_ttft: float = 0.0            # 当前平均首token时间（秒）
    avg_latency: float = 0.0         # 当前平均延迟（秒）


@dataclass
class GPUMonitorResults:
    """GPU监控结果汇总
    
    包含整个benchmark过程中的所有GPU快照数据，
    并提供计算能耗、平均功率等衍生指标的方法。
    """
    gpu_ids: List[int] = field(default_factory=list)      # 监控的GPU列表
    snapshots: List[GPUSnapshot] = field(default_factory=list)  # 所有采样快照
    start_time: float = 0.0  # 监控开始时间（Unix时间戳）
    end_time: float = 0.0    # 监控结束时间（Unix时间戳）
    
    # 最终性能指标（在benchmark结束时设置）
    total_tokens_generated: int = 0    # 总共生成的token数
    total_tokens_accepted: int = 0     # 总共接受的token数（推测解码）
    total_requests: int = 0            # 总共完成的请求数
    
    @property
    def duration(self) -> float:
        """监控总时长（秒）"""
        return self.end_time - self.start_time
    
    @property
    def total_energy_consumed(self) -> Dict[int, float]:
        """计算每个GPU的总能耗（焦耳）
        
        使用梯形积分法计算能耗：
        E = ∫ P(t) dt ≈ Σ [(P[i] + P[i+1]) / 2 * Δt]
        
        算法流程：
        1. 按GPU分组所有快照
        2. 按时间排序
        3. 对相邻两个采样点，计算：
           - 时间间隔 Δt = t[i+1] - t[i]
           - 平均功率 P_avg = (P[i] + P[i+1]) / 2
           - 能量增量 ΔE = P_avg * Δt
        4. 累加所有时间段的能量
        
        返回：
            Dict[gpu_id, energy_joules]: 每个GPU的总能耗（焦耳）
        
        示例：
            {0: 150.5, 1: 148.2, ...}  # GPU 0消耗150.5焦耳
        """
        energy = defaultdict(float)
        
        # 按GPU分组快照
        gpu_snapshots = defaultdict(list)
        for snapshot in self.snapshots:
            gpu_snapshots[snapshot.gpu_id].append(snapshot)
        
        # 对每个GPU计算能耗
        for gpu_id, snapshots in gpu_snapshots.items():
            if len(snapshots) < 2:
                continue  # 少于2个采样点无法积分
            
            # 按时间排序
            snapshots.sort(key=lambda x: x.timestamp)
            
            # 梯形积分：累加每个时间段的能量
            for i in range(len(snapshots) - 1):
                dt = snapshots[i + 1].timestamp - snapshots[i].timestamp  # 时间间隔（秒）
                avg_power = (snapshots[i].power_draw + snapshots[i + 1].power_draw) / 2  # 平均功率（瓦特）
                energy[gpu_id] += avg_power * dt  # 能量 = 功率 × 时间（焦耳）
        
        return dict(energy)
    
    @property
    def average_power(self) -> Dict[int, float]:
        """Average power consumption per GPU in Watts."""
        power_sum = defaultdict(float)
        power_count = defaultdict(int)
        
        for snapshot in self.snapshots:
            power_sum[snapshot.gpu_id] += snapshot.power_draw
            power_count[snapshot.gpu_id] += 1
        
        return {
            gpu_id: power_sum[gpu_id] / power_count[gpu_id] 
            if power_count[gpu_id] > 0 else 0.0
            for gpu_id in power_sum.keys()
        }
    
    @property
    def peak_power(self) -> Dict[int, float]:
        """Peak power consumption per GPU in Watts."""
        peak = defaultdict(float)
        for snapshot in self.snapshots:
            peak[snapshot.gpu_id] = max(peak[snapshot.gpu_id], snapshot.power_draw)
        return dict(peak)
    
    @property
    def average_utilization(self) -> Dict[int, float]:
        """Average GPU utilization per GPU in percentage."""
        util_sum = defaultdict(float)
        util_count = defaultdict(int)
        
        for snapshot in self.snapshots:
            util_sum[snapshot.gpu_id] += snapshot.utilization_gpu
            util_count[snapshot.gpu_id] += 1
        
        return {
            gpu_id: util_sum[gpu_id] / util_count[gpu_id] 
            if util_count[gpu_id] > 0 else 0.0
            for gpu_id in util_sum.keys()
        }
    
    @property
    def average_memory_usage(self) -> Dict[int, float]:
        """Average memory usage per GPU in percentage."""
        mem_sum = defaultdict(float)
        mem_count = defaultdict(int)
        
        for snapshot in self.snapshots:
            mem_pct = (snapshot.memory_used / snapshot.memory_total * 100) if snapshot.memory_total > 0 else 0
            mem_sum[snapshot.gpu_id] += mem_pct
            mem_count[snapshot.gpu_id] += 1
        
        return {
            gpu_id: mem_sum[gpu_id] / mem_count[gpu_id] 
            if mem_count[gpu_id] > 0 else 0.0
            for gpu_id in mem_sum.keys()
        }
    
    @property
    def peak_temperature(self) -> Dict[int, float]:
        """Peak temperature per GPU in Celsius."""
        temp = defaultdict(float)
        for snapshot in self.snapshots:
            temp[snapshot.gpu_id] = max(temp[snapshot.gpu_id], snapshot.temperature)
        return dict(temp)
    
    @property
    def total_energy_all_gpus(self) -> float:
        """Total energy consumed across all GPUs in Joules."""
        return sum(self.total_energy_consumed.values())
    
    @property
    def tokens_per_joule(self) -> float:
        """每焦耳生成的token数（能效指标）
        
        计算公式：tokens_per_joule = 总token数 / 总能耗(焦耳)
        
        这是衡量推理能效的核心指标，数值越高表示能效越好。
        
        返回：
            float: 每焦耳生成的token数（tokens/J）
        
        示例：
            2.5 tokens/J 表示消耗1焦耳能量可以生成2.5个token
        """
        if self.total_energy_all_gpus <= 0:
            return 0.0
        return self.total_tokens_generated / self.total_energy_all_gpus
    
    @property
    def tokens_accepted_per_joule(self) -> float:
        """每焦耳接受的token数（推测解码专用能效指标）
        
        仅用于推测解码模式，计算被target模型接受的token与能耗的比率。
        接受的token数 < 生成的token数（因为有些draft token会被拒绝）。
        
        计算公式：tokens_accepted_per_joule = 接受的token数 / 总能耗(焦耳)
        
        返回：
            float: 每焦耳接受的token数（tokens/J）
        """
        if self.total_energy_all_gpus <= 0:
            return 0.0
        if self.total_tokens_accepted > 0:
            return self.total_tokens_accepted / self.total_energy_all_gpus
        # 如果没有接受数据（非推测解码模式），回退到生成数
        return self.total_tokens_generated / self.total_energy_all_gpus
    
    @property
    def tokens_per_kwh(self) -> float:
        """每千瓦时生成的token数（工业化能效指标）
        
        将能耗转换为更直观的千瓦时(kWh)单位，便于与电费挂钩。
        
        转换关系：
        1 kWh = 3,600,000 J (1千瓦时 = 1000瓦 × 3600秒)
        
        计算公式：
        1. 能耗(kWh) = 总能耗(焦耳) / 3,600,000
        2. tokens_per_kwh = 总token数 / 能耗(kWh)
        
        返回：
            float: 每千瓦时生成的token数（tokens/kWh）
        
        实际应用示例：
            tokens_per_kwh = 1,000,000
            电费单价 = 0.6元/kWh
            → 生成100万token需要1度电，成本0.6元
        """
        energy_kwh = self.total_energy_all_gpus / 3600000  # 焦耳转千瓦时
        if energy_kwh <= 0:
            return 0.0
        return self.total_tokens_generated / energy_kwh
    
    @property
    def tokens_accepted_per_kwh(self) -> float:
        """每千瓦时接受的token数（推测解码专用，工业化指标）
        
        结合推测解码和工业化单位，用于评估推测解码的实际能效收益。
        
        计算公式：
        1. 能耗(kWh) = 总能耗(焦耳) / 3,600,000
        2. tokens_accepted_per_kwh = 接受的token数 / 能耗(kWh)
        
        返回：
            float: 每千瓦时接受的token数（tokens/kWh）
        
        对比意义：
        - tokens_per_kwh: 衡量"生成"效率（包括被拒绝的draft）
        - tokens_accepted_per_kwh: 衡量"有效产出"效率（仅被接受的）
        """
        energy_kwh = self.total_energy_all_gpus / 3600000  # 焦耳转千瓦时
        if energy_kwh <= 0:
            return 0.0
        if self.total_tokens_accepted > 0:
            return self.total_tokens_accepted / energy_kwh
        return self.total_tokens_generated / energy_kwh
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "gpu_ids": self.gpu_ids,
            "duration": self.duration,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_energy_consumed": {str(k): v for k, v in self.total_energy_consumed.items()},
            "total_energy_all_gpus": self.total_energy_all_gpus,
            "average_power": {str(k): v for k, v in self.average_power.items()},
            "peak_power": {str(k): v for k, v in self.peak_power.items()},
            "average_utilization": {str(k): v for k, v in self.average_utilization.items()},
            "average_memory_usage": {str(k): v for k, v in self.average_memory_usage.items()},
            "peak_temperature": {str(k): v for k, v in self.peak_temperature.items()},
            "total_tokens_generated": self.total_tokens_generated,
            "total_tokens_accepted": self.total_tokens_accepted,
            "total_requests": self.total_requests,
            "tokens_per_joule": self.tokens_per_joule,
            "tokens_accepted_per_joule": self.tokens_accepted_per_joule,
            "tokens_per_kwh": self.tokens_per_kwh,
            "tokens_accepted_per_kwh": self.tokens_accepted_per_kwh,
            "snapshots": [asdict(s) for s in self.snapshots]
        }


class GPUMonitor:
    """Monitors GPU power and performance metrics."""
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, sampling_interval: float = 1.0, 
                 performance_callback: Optional[callable] = None):
        """
        Initialize GPU monitor.
        
        Args:
            gpu_ids: List of GPU IDs to monitor. If None, monitors all available GPUs.
            sampling_interval: Sampling interval in seconds (default: 1.0).
            performance_callback: Optional callback function that returns performance metrics dict:
                {
                    'total_tokens_generated': int,
                    'total_tokens_accepted': int,
                    'requests_completed': int,
                    'throughput': float,
                    'avg_ttft': float,
                    'avg_latency': float
                }
        """
        self.gpu_ids = gpu_ids if gpu_ids is not None else self._detect_gpu_ids()
        self.sampling_interval = sampling_interval
        self.performance_callback = performance_callback
        self.results = GPUMonitorResults(gpu_ids=self.gpu_ids)
        self._monitoring = False
        self._monitor_thread = None
        
    def _detect_gpu_ids(self) -> List[int]:
        """Detect available GPU IDs."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_ids = [int(line.strip()) for line in result.stdout.strip().split('\n') if line.strip()]
            return gpu_ids
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(colored(f"⚠️  Warning: Could not detect GPUs: {e}", "yellow"))
            return []
    
    def _query_gpu_metrics(self, gpu_id: int) -> Optional[GPUSnapshot]:
        """Query metrics for a single GPU."""
        try:
            query = (
                "index,power.draw,power.limit,utilization.gpu,utilization.memory,"
                "memory.used,memory.total,temperature.gpu,clocks.current.graphics,"
                "clocks.current.memory"
            )
            
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_id}",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            
            parts = result.stdout.strip().split(', ')
            if len(parts) < 10:
                return None
            
            timestamp = time.time()
            return GPUSnapshot(
                timestamp=timestamp,
                gpu_id=int(parts[0]),
                power_draw=float(parts[1]) if parts[1] != '[Not Supported]' else 0.0,
                power_limit=float(parts[2]) if parts[2] != '[Not Supported]' else 0.0,
                utilization_gpu=float(parts[3]) if parts[3] != '[Not Supported]' else 0.0,
                utilization_memory=float(parts[4]) if parts[4] != '[Not Supported]' else 0.0,
                memory_used=int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                memory_total=int(parts[6]) if parts[6] != '[Not Supported]' else 0,
                temperature=float(parts[7]) if parts[7] != '[Not Supported]' else 0.0,
                clock_graphics=int(parts[8]) if parts[8] != '[Not Supported]' else 0,
                clock_memory=int(parts[9]) if parts[9] != '[Not Supported]' else 0,
            )
        except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
            print(colored(f"⚠️  Warning: Could not query GPU {gpu_id}: {e}", "yellow"))
            return None
    
    def _monitor_loop(self):
        """Main monitoring loop running in background thread."""
        while self._monitoring:
            snapshot_time = time.time()
            
            # Get performance metrics if callback is available
            perf_metrics = {}
            if self.performance_callback:
                try:
                    perf_metrics = self.performance_callback() or {}
                except Exception as e:
                    print(colored(f"⚠️  Warning: Performance callback error: {e}", "yellow"))
            
            for gpu_id in self.gpu_ids:
                snapshot = self._query_gpu_metrics(gpu_id)
                if snapshot:
                    # Add performance metrics to snapshot
                    snapshot.total_tokens_generated = perf_metrics.get('total_tokens_generated', 0)
                    snapshot.total_tokens_accepted = perf_metrics.get('total_tokens_accepted', 0)
                    snapshot.requests_completed = perf_metrics.get('requests_completed', 0)
                    snapshot.throughput = perf_metrics.get('throughput', 0.0)
                    snapshot.avg_ttft = perf_metrics.get('avg_ttft', 0.0)
                    snapshot.avg_latency = perf_metrics.get('avg_latency', 0.0)
                    
                    self.results.snapshots.append(snapshot)
            
            # Sleep until next sampling interval
            elapsed = time.time() - snapshot_time
            sleep_time = max(0, self.sampling_interval - elapsed)
            time.sleep(sleep_time)
    
    def start(self):
        """Start GPU monitoring."""
        if self._monitoring:
            print(colored("⚠️  Warning: Monitor already running", "yellow"))
            return
        
        self.results = GPUMonitorResults(gpu_ids=self.gpu_ids)
        self.results.start_time = time.time()
        self._monitoring = True
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(colored(f"✅ GPU Monitor started (GPUs: {self.gpu_ids}, interval: {self.sampling_interval}s)", "green"))
    
    def stop(self):
        """Stop GPU monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.results.end_time = time.time()
        print(colored("✅ GPU Monitor stopped", "green"))
    
    def get_results(self) -> GPUMonitorResults:
        """Get monitoring results."""
        return self.results
    
    def save_results(self, filepath: str, results: Optional[GPUMonitorResults] = None):
        """Save results to JSON file."""
        results_to_save = results if results is not None else self.results
        with open(filepath, 'w') as f:
            json.dump(results_to_save.to_dict(), f, indent=2)
        print(colored(f"✅ GPU monitoring results saved to {filepath}", "green"))


def print_gpu_summary(results: GPUMonitorResults):
    """Print formatted summary of GPU monitoring results."""
    print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))
    print(colored("📊 GPU Power & Performance Summary", "cyan", attrs=["bold"]))
    print(colored("=" * 70, "cyan", attrs=["bold"]))
    
    print(colored("\n⚡ Power Consumption:", "yellow", attrs=["bold"]))
    total_energy = 0.0
    total_avg_power = 0.0
    
    for gpu_id in results.gpu_ids:
        energy = results.total_energy_consumed.get(gpu_id, 0.0)
        avg_power = results.average_power.get(gpu_id, 0.0)
        peak_power = results.peak_power.get(gpu_id, 0.0)
        
        energy_wh = energy / 3600  # Convert Joules to Wh
        
        print(f"  GPU {gpu_id}:")
        print(f"    Average Power:     {avg_power:.2f} W")
        print(f"    Peak Power:        {peak_power:.2f} W")
        print(f"    Total Energy:      {energy_wh:.2f} Wh ({energy:.2f} J)")
        
        total_energy += energy
        total_avg_power += avg_power
    
    print(f"\n  Total (All GPUs):")
    print(f"    Average Power:     {total_avg_power:.2f} W")
    energy_wh_total = total_energy / 3600
    energy_kwh_total = energy_wh_total / 1000
    print(f"    Total Energy:      {energy_wh_total:.2f} Wh ({energy_kwh_total:.4f} kWh)")
    
    print(colored("\n📈 Utilization:", "yellow", attrs=["bold"]))
    for gpu_id in results.gpu_ids:
        util = results.average_utilization.get(gpu_id, 0.0)
        mem_usage = results.average_memory_usage.get(gpu_id, 0.0)
        print(f"  GPU {gpu_id}:")
        print(f"    GPU Utilization:  {util:.1f}%")
        print(f"    Memory Usage:      {mem_usage:.1f}%")
    
    print(colored("\n🌡️  Temperature:", "yellow", attrs=["bold"]))
    for gpu_id in results.gpu_ids:
        temp = results.peak_temperature.get(gpu_id, 0.0)
        print(f"  GPU {gpu_id}: Peak Temperature: {temp:.1f}°C")
    
    if results.total_tokens_generated > 0:
        print(colored("\n🎯 Performance Metrics:", "yellow", attrs=["bold"]))
        print(f"  Total Tokens Generated: {results.total_tokens_generated:,}")
        if results.total_tokens_accepted > 0:
            print(f"  Total Tokens Accepted:   {results.total_tokens_accepted:,}")
        print(f"  Total Requests:         {results.total_requests}")
        
        print(colored("\n⚡ Energy Efficiency:", "yellow", attrs=["bold"]))
        print(f"  Tokens per Joule:       {results.tokens_per_joule:.2f} tokens/J")
        print(f"  Tokens per kWh:         {results.tokens_per_kwh:,.0f} tokens/kWh")
        if results.total_tokens_accepted > 0:
            print(f"  Accepted Tokens per Joule: {results.tokens_accepted_per_joule:.2f} tokens/J")
            print(f"  Accepted Tokens per kWh:   {results.tokens_accepted_per_kwh:,.0f} tokens/kWh")
    
    print(colored(f"\n⏱️  Monitoring Duration: {results.duration:.2f} s", "yellow", attrs=["bold"]))
    print(colored("=" * 70, "cyan", attrs=["bold"]))

