"""
Configuration management for different run modes and parameters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ServerConfig:
    """Configuration for llama.cpp server connection."""
    url: str = "http://127.0.0.1:8080/completion"
    timeout_s: float = 20.0
    connect_timeout_s: float = 3.0


@dataclass
class GrammarConfig:
    """Grammar file configuration."""
    num_grammar_file: str = "grammars/final/int_strict_final.gbnf"
    yesno_grammar_file: str = "grammars/final/yesno_strict_final.gbnf"

    def load_grammar(self, filepath: str) -> str:
        """Load grammar file content."""
        try:
            return Path(filepath).read_text(encoding="utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Grammar file not found: {filepath}")

    def get_num_grammar(self) -> str:
        """Load numeric grammar."""
        return self.load_grammar(self.num_grammar_file)

    def get_yesno_grammar(self) -> str:
        """Load yes/no grammar."""
        return self.load_grammar(self.yesno_grammar_file)


@dataclass
class InferenceConfig:
    """Inference parameters."""
    n_predict_num: int = 32  # Max tokens for numeric answers
    n_predict_log: int = 8   # Max tokens for yes/no answers
    temperature: float = 0.0  # Always 0 for deterministic results
    repeats: int = 3  # Number of trials per prompt
    warmup_per_prompt: int = 0  # Warmup inferences per prompt

    def get_n_predict(self, category: str) -> int:
        """Get n_predict based on category."""
        return self.n_predict_log if category.upper() == "LOG" else self.n_predict_num


@dataclass
class RunConfig:
    """Complete configuration for a run."""
    mode: str = "baseline"  # baseline, hybrid-v1, hybrid-v2
    use_grammar: bool = True  # Use grammar constraints
    server: ServerConfig = None
    grammar: GrammarConfig = None
    inference: InferenceConfig = None
    verbose: bool = False
    print_every: int = 1  # Print progress every N prompts
    show_prompt: bool = False  # Show prompt text in verbose mode
    debug: bool = False  # Enable debug output

    def __post_init__(self):
        if self.server is None:
            self.server = ServerConfig()
        if self.grammar is None:
            self.grammar = GrammarConfig()
        if self.inference is None:
            self.inference = InferenceConfig()

    @property
    def variant_name(self) -> str:
        """Generate variant name for results."""
        base = self.mode
        if self.mode == "baseline":
            base = "phi2_server"
        suffix = "grammar" if self.use_grammar else "nogrammar"
        return f"{base}_{suffix}"


def create_baseline_config(
    use_grammar: bool = True,
    timeout_s: float = 20.0,
    repeats: int = 3,
    verbose: bool = False,
    debug: bool = False,
) -> RunConfig:
    """Create configuration for baseline runs."""
    return RunConfig(
        mode="baseline",
        use_grammar=use_grammar,
        server=ServerConfig(timeout_s=timeout_s),
        inference=InferenceConfig(repeats=repeats),
        verbose=verbose,
        debug=debug,
    )


def create_hybrid_v1_config(
    timeout_s: float = 90.0,
    repeats: int = 1,
    verbose: bool = False,
) -> RunConfig:
    """Create configuration for hybrid v1 runs (SymPy routing)."""
    return RunConfig(
        mode="hybrid-v1",
        use_grammar=False,  # Hybrid v1 doesn't use grammar
        server=ServerConfig(timeout_s=timeout_s),
        inference=InferenceConfig(repeats=repeats),
        verbose=verbose,
    )


def create_hybrid_v2_config(
    timeout_s: float = 120.0,
    repeats: int = 1,
    use_yesno_grammar: bool = True,
    verbose: bool = False,
) -> RunConfig:
    """Create configuration for hybrid v2 runs (LLM + SymPy verify)."""
    return RunConfig(
        mode="hybrid-v2",
        use_grammar=use_yesno_grammar,  # Only for LOG category
        server=ServerConfig(timeout_s=timeout_s),
        inference=InferenceConfig(repeats=repeats),
        verbose=verbose,
    )
