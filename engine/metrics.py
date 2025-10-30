"""Performance metrics collection and reporting for Speculative Decoding benchmarks."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import json
from termcolor import colored


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    
    # Time metrics (in seconds)
    ttft: float = 0.0  # Time To First Token
    time_per_token: List[float] = field(default_factory=list)  # Per-token latency
    total_latency: float = 0.0  # End-to-end latency
    
    # Speculative decoding specific
    acceptance_rate: float = 0.0
    drafts_generated: int = 0
    drafts_accepted: int = 0
    
    # Timestamps
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0


@dataclass
class BatchMetrics:
    """Metrics for a batch of requests."""
    batch_size: int = 0
    requests: List[RequestMetrics] = field(default_factory=list)
    
    batch_start_time: float = 0.0
    batch_end_time: float = 0.0
    
    @property
    def batch_latency(self) -> float:
        """Total batch processing time."""
        return self.batch_end_time - self.batch_start_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens generated in batch."""
        return sum(r.generated_tokens for r in self.requests)
    
    @property
    def avg_ttft(self) -> float:
        """Average TTFT across batch."""
        if not self.requests:
            return 0.0
        return sum(r.ttft for r in self.requests) / len(self.requests)
    
    @property
    def avg_latency(self) -> float:
        """Average latency across batch."""
        if not self.requests:
            return 0.0
        return sum(r.total_latency for r in self.requests) / len(self.requests)
    
    @property
    def throughput(self) -> float:
        """Tokens per second for this batch."""
        if self.batch_latency <= 0:
            return 0.0
        return self.total_tokens / self.batch_latency


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""
    method: str  # "speculative" or "target_ar"
    total_requests: int = 0
    total_batches: int = 0
    batches: List[BatchMetrics] = field(default_factory=list)
    
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def total_duration(self) -> float:
        """Total benchmark duration."""
        return self.end_time - self.start_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens generated."""
        return sum(b.total_tokens for b in self.batches)
    
    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens."""
        return sum(r.prompt_tokens for b in self.batches for r in b.requests)
    
    @property
    def overall_throughput(self) -> float:
        """Overall throughput (tokens/second)."""
        if self.total_duration <= 0:
            return 0.0
        return self.total_tokens / self.total_duration
    
    @property
    def avg_ttft(self) -> float:
        """Average TTFT across all requests."""
        all_requests = [r for b in self.batches for r in b.requests]
        if not all_requests:
            return 0.0
        return sum(r.ttft for r in all_requests) / len(all_requests)
    
    @property
    def avg_latency(self) -> float:
        """Average latency across all requests."""
        all_requests = [r for b in self.batches for r in b.requests]
        if not all_requests:
            return 0.0
        return sum(r.total_latency for r in all_requests) / len(all_requests)
    
    @property
    def avg_acceptance_rate(self) -> float:
        """Average acceptance rate (for speculative decoding)."""
        all_requests = [r for b in self.batches for r in b.requests if r.acceptance_rate > 0]
        if not all_requests:
            return 0.0
        return sum(r.acceptance_rate for r in all_requests) / len(all_requests)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method,
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "overall_throughput": self.overall_throughput,
            "avg_ttft": self.avg_ttft,
            "avg_latency": self.avg_latency,
            "avg_acceptance_rate": self.avg_acceptance_rate,
            "batches": [
                {
                    "batch_size": b.batch_size,
                    "batch_latency": b.batch_latency,
                    "total_tokens": b.total_tokens,
                    "avg_ttft": b.avg_ttft,
                    "avg_latency": b.avg_latency,
                    "throughput": b.throughput,
                    "requests": [
                        {
                            "prompt_tokens": r.prompt_tokens,
                            "generated_tokens": r.generated_tokens,
                            "total_tokens": r.total_tokens,
                            "ttft": r.ttft,
                            "total_latency": r.total_latency,
                            "acceptance_rate": r.acceptance_rate,
                            "drafts_generated": r.drafts_generated,
                            "drafts_accepted": r.drafts_accepted,
                        }
                        for r in b.requests
                    ]
                }
                for b in self.batches
            ]
        }
    
    def save_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(colored(f"✅ Results saved to {filepath}", "green"))


def print_benchmark_summary(results: BenchmarkResults):
    """Print a formatted summary of benchmark results."""
    print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))
    print(colored(f"📊 Benchmark Results: {results.method.upper()}", "cyan", attrs=["bold"]))
    print(colored("=" * 70, "cyan", attrs=["bold"]))
    
    print(colored("\n🎯 Overall Statistics:", "yellow", attrs=["bold"]))
    print(f"  Total Requests:     {results.total_requests}")
    print(f"  Total Batches:      {results.total_batches}")
    print(f"  Total Duration:     {results.total_duration:.2f} s")
    print(f"  Total Tokens:       {results.total_tokens:,}")
    print(f"  Prompt Tokens:      {results.total_prompt_tokens:,}")
    print(f"  Generated Tokens:   {results.total_tokens - results.total_prompt_tokens:,}")
    
    print(colored("\n⚡ Performance Metrics:", "yellow", attrs=["bold"]))
    print(f"  Overall Throughput: {results.overall_throughput:.2f} tokens/s")
    print(f"  Average TTFT:       {results.avg_ttft*1000:.2f} ms")
    print(f"  Average Latency:     {results.avg_latency*1000:.2f} ms")
    
    if results.method == "speculative":
        print(colored("\n🎲 Speculative Decoding Metrics:", "yellow", attrs=["bold"]))
        print(f"  Average Acceptance Rate: {results.avg_acceptance_rate:.3f}")
    
    print(colored("\n" + "=" * 70, "cyan", attrs=["bold"]))


def print_comparison(spec_results: BenchmarkResults, target_results: BenchmarkResults):
    """Print comparison between speculative and target AR results."""
    print(colored("\n" + "=" * 70, "magenta", attrs=["bold"]))
    print(colored("📈 Performance Comparison", "magenta", attrs=["bold"]))
    print(colored("=" * 70, "magenta", attrs=["bold"]))
    
    speedup = target_results.avg_latency / spec_results.avg_latency if spec_results.avg_latency > 0 else 0
    throughput_gain = (spec_results.overall_throughput / target_results.overall_throughput - 1) * 100 if target_results.overall_throughput > 0 else 0
    
    print(colored("\n⚡ Speed Metrics:", "yellow", attrs=["bold"]))
    print(f"  Throughput Speedup:  {speedup:.2f}x")
    print(f"  Throughput Gain:     {throughput_gain:+.1f}%")
    print(f"  Latency Reduction:   {(1 - spec_results.avg_latency / target_results.avg_latency) * 100:.1f}%" if target_results.avg_latency > 0 else "  Latency Reduction:   N/A")
    
    print(colored("\n📊 Detailed Comparison:", "yellow", attrs=["bold"]))
    print(f"{'Metric':<25} {'Speculative':<15} {'Target AR':<15} {'Ratio':<10}")
    print("-" * 70)
    print(f"{'Throughput (tok/s)':<25} {spec_results.overall_throughput:<15.2f} {target_results.overall_throughput:<15.2f} {speedup:<10.2f}x")
    print(f"{'Avg TTFT (ms)':<25} {spec_results.avg_ttft*1000:<15.2f} {target_results.avg_ttft*1000:<15.2f} {spec_results.avg_ttft/target_results.avg_ttft:<10.2f}x" if target_results.avg_ttft > 0 else f"{'Avg TTFT (ms)':<25} {spec_results.avg_ttft*1000:<15.2f} {target_results.avg_ttft*1000:<15.2f} {'N/A':<10}")
    print(f"{'Avg Latency (ms)':<25} {spec_results.avg_latency*1000:<15.2f} {target_results.avg_latency*1000:<15.2f} {speedup:<10.2f}x")
    
    print(colored("\n" + "=" * 70, "magenta", attrs=["bold"]))

