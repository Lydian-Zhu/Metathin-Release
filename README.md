# Metathin

**A Meta-cognitive Agent System Construction Framework**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Lydian-Zhu/Metathin-Release/pulls)

---

## What is Metathin?

Metathin is a framework for building cognitive agents based on a five-tuple structure `(P, B, S, D, Ψ)`. It provides a clean, modular, and extensible foundation for creating intelligent systems.

**Design Philosophy:**  
- **Fixed Interface, Free Implementation** – You bring your own algorithms  
- **Modular & Composable** – Plug and play components  
- **Type Safe & Observable** – Built with type hints and full logging

I started this project for my physics experiment data processing, and later realized it could become something more general. Now it's here, hope it helps.

---

## The Core Idea

No matter what kind of agent you're building—a chatbot, a lab assistant, or a prediction model—its thinking process can be described with five elements:
Metathin = (P, B, S, D, Ψ)

text

- **P (PatternSpace)** – Perception: turns raw input into feature vectors  
- **B (MetaBehavior)** – Action: executable skill units  
- **S (Selector)** – Evaluation: computes how suitable each behavior is  
- **D (DecisionStrategy)** – Decision: picks the best behavior  
- **Ψ (LearningMechanism)** – Learning: adjusts parameters from feedback  

This abstraction doesn't care about what AI technology you use underneath. It just works.

---

## Built-in Components? Optional.

Metathin comes with 30+ pre-built components (`SimplePatternSpace`, `MaxFitnessStrategy`, `GradientLearning`...). **But you don't have to use them.**

The framework's real value is in the **interfaces**:

```python
from metathin.core import PatternSpace, MetaBehavior, Selector, DecisionStrategy, LearningMechanism
You can:

Use what I provide (quick start)

Implement your own algorithms (full control)

Mix and match (bring your own Selector, use my Behaviors)

The interfaces are designed to be minimal and flexible. They don't force you into any specific implementation. You bring your domain expertise, the framework handles the orchestration.

AI? Feel Free to Ask It
The project structure is clean, the interfaces are well-documented, and the type hints are everywhere. If you paste this code into an AI assistant and say "Help me implement a custom Selector for my problem", it'll probably know what to do. I've designed it that way on purpose.

Project Structure (Refactored)
text
metathin/                    # Core framework
├── core/                   # Five-element interfaces (P, B, S, D, Ψ)
│   ├── p_pattern.py       # PatternSpace interface
│   ├── b_behavior.py      # MetaBehavior interface
│   ├── s_selector.py      # Selector interface
│   ├── d_decision.py      # DecisionStrategy interface
│   ├── psi_learning.py    # LearningMechanism interface
│   └── memory_backend.py  # Storage backend interface
├── engine/                 # Thinking pipeline
│   ├── pipeline.py        # Pure function cognitive cycle
│   ├── context.py         # State container
│   └── hooks.py           # Extension points
├── services/               # Optional services
│   ├── memory_manager.py  # Two-tier caching
│   ├── history_tracker.py # Thought history
│   └── metrics_collector.py # Performance metrics
├── config/                 # Configuration system
│   ├── schema.py          # Config data structures
│   └── loader.py          # Load from file/env
├── agent/                  # Facade & builder
│   ├── metathin.py        # Main agent class
│   └── builder.py         # Fluent builder
└── components/             # Built-in implementations
    ├── pattern_space.py   # 5+ feature extractors
    ├── behavior_library.py # 7+ behavior wrappers
    ├── selector.py        # 5+ fitness calculators
    ├── decision.py        # 7+ decision strategies
    └── learning.py        # 5+ learning mechanisms

metathin_plus/              # Domain extensions
└── sci/                   # Scientific discovery
Each file name tells you what it does – p_pattern.py = Pattern Space, b_behavior.py = Behavior, etc.

What Makes Metathin Different?
Memory Built-in, Not Bolted-on
Most frameworks treat memory as an afterthought. In Metathin, it's built into the core from day one.

Every agent comes with a complete memory system:

Two-tier architecture: Fast in-memory cache + persistent backend

Multiple backends: JSON (human-readable), SQLite (production), in-memory (testing) – swap them anytime

Smart management: LRU eviction when cache fills up, TTL (time-to-live) for auto-expiring items

Every component can use it: Behaviors can remember outcomes, Selectors can recall past fitness, Learning mechanisms can store experiences

RL-ready, but not RL-only
The five-tuple structure maps naturally to reinforcement learning:

P = state representation

B = action space

S = value function

D = policy

Ψ = update rule

But swap in a supervised Selector and a gradient-based Learner, and you're doing supervised learning. Swap in Hebbian learning with no expected output, and you're doing unsupervised learning.

Same architecture. Different components. Three paradigms.

Clean, Layered Architecture
The refactored codebase is organized into clear layers:

Core – Pure interfaces, no dependencies

Engine – Pure function pipeline, stateless

Services – Optional, injectable

Config – Separate from logic

Agent – Facade that assembles everything

Components – Concrete implementations

This makes the code easy to understand, test, and extend.

Quick Start
python
from metathin import Metathin, MetathinBuilder
from metathin.components import SimplePatternSpace, FunctionBehavior, MaxFitnessStrategy

# Define behaviors
greet = FunctionBehavior("greet", lambda f,**k: "Hello! I'm an agent, how can I help?")
echo = FunctionBehavior("echo", lambda f,**k: f"You just said: {k.get('user_input', '')}")

# Build agent (builder pattern)
agent = (MetathinBuilder()
    .with_pattern_space(SimplePatternSpace(lambda x: [len(x)]))
    .with_behaviors([greet, echo])
    .with_decision_strategy(MaxFitnessStrategy())
    .with_name("SimpleAgent")
    .build())

# Or use direct constructor
agent = Metathin(
    pattern_space=SimplePatternSpace(lambda x: [len(x)]),
    decision_strategy=MaxFitnessStrategy(),
    name="SimpleAgent"
)
agent.register_behaviors([greet, echo])

# Chat
print("Agent started, type 'quit' to exit")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == 'quit':
        break
    
    response = agent.think(user_input, user_input=user_input)
    print(f"Agent: {response}")
Run it. It works.

Configuration Examples
python
from metathin.config import MetathinConfig, MemoryConfig, ObservabilityConfig

# Minimal config (no memory, no history)
config = MetathinConfig.create_minimal()

# Production config (SQLite memory, full observability)
config = MetathinConfig.create_production("MyAgent")

# Custom config
config = MetathinConfig(
    memory=MemoryConfig(enabled=True, backend_type='sqlite'),
    observability=ObservabilityConfig(keep_history=True, enable_metrics=True)
)
Built-in Modules (To Get You Started)
Scientific Discovery (metathin_plus.sci)
For finding patterns and laws from data:

Symbolic regression engine

Feature extraction (30+ features)

Similarity matching with pre-trained function libraries

Adaptive extrapolation

If you work with experimental data—chemistry, biology, materials science—this might save you time.

The Sci Module? Also Customizable
metathin_plus.sci follows the same philosophy:

Use the built-in function generator, feature extractor, and similarity matcher

OR implement your own discovery algorithms

OR extend the existing ones

The memory system (FunctionMemoryBank) is pluggable. The extrapolator is adaptable. Nothing is locked down.

This is your framework. Use it how you want.

This is Where You Come In
Metathin's core is intentionally minimal. The real power comes from you adding modules for your own field.

If You're in Physics/Chemistry/Biology
Add to metathin_plus.sci:

New scientific discovery algorithms

Domain-specific function libraries

Pattern discovery examples from your own data

If You're in Economics
Create metathin_plus.econ:

Market prediction models

Agent-based economic simulations

Time series forecasting for financial data

If You're in Neuroscience
Create metathin_plus.neuro:

Neural spike train analysis

Brain-computer interface components

Cognitive modeling tools

If You're in Climate Science
Create metathin_plus.climate:

Weather prediction models

Climate pattern analysis

Ensemble forecasting

You get the idea. Any domain that involves data, patterns, and decision-making can build on Metathin.

Why Contribute?
Your work gets used – by researchers, students, and practitioners in your field

You stand on shoulders – the core framework handles the boring stuff, you focus on domain logic

It's good for your CV – open source contributions matter

It's actually fun – seeing your code help other people is satisfying

How to Add Your Own Module
Fork the repo

Create metathin_plus/your_module/

Implement your components (they just need to follow the core interfaces)

Add a few examples so people know how to use it

Open a PR

That's it. No permission needed. If your module is useful, it gets merged. If it's experimental, we can mark it as such. The goal is to let people share.

Current Status (v0.4.0)
What works:

✅ Core five-element architecture (refactored)

✅ All built-in components (30+)

✅ Memory system (JSON/SQLite/in-memory backends, TTL, LRU)

✅ Thinking pipeline with hooks

✅ Configuration system (file/env/dict)

✅ Builder pattern for easy agent creation

✅ Scientific discovery module

✅ 492+ unit tests passing

Under the hood, Metathin already supports:

✅ Supervised learning (via GradientLearning + expected)

✅ Unsupervised learning (via HebbianLearning)

✅ Reinforcement learning (via RewardLearning + exploration strategies)

✅ Time series forecasting (with built-in memory for history)

What's in progress:

⚠️ User manual is being written

⚠️ More examples and tutorials

⚠️ Performance optimizations

Why not on PyPI yet?
I want to get more feedback first, make sure the API feels right. Once stable, it'll be there.

Installation
bash
# Core only
pip install git+https://github.com/Lydian-Zhu/Metathin-Release.git

# With scientific discovery
pip install git+https://github.com/Lydian-Zhu/Metathin-Release.git#egg=metathin[sci]

# Everything
pip install git+https://github.com/Lydian-Zhu/Metathin-Release.git#egg=metathin[full]
Dependencies are grouped, so you only install what you need.

How to Contribute (Even Without Code)
Star the repo – lets me know someone cares

Open an issue – found a bug? have an idea?

Improve docs – fix typos, add explanations

Write examples – show how you use it in your field

Tell people – share it with your lab, your students, your Twitter followers

The Vision
A growing collection of domain-specific modules, all built on the same core framework. A physicist's tools, a chemist's reaction kinetics, an economist's market models—all accessible through the same clean API.

Is that ambitious? Yes.
Can we build it together? Also yes.

License
MIT – do what you want, just don't blame me if it breaks.

Acknowledgements
Started from my own experimental data needs. If you find it useful, great. If you want to build it together, even better.

If you like it, star it. If you want to help, PR it. If you have ideas, issue it. If you want to add your own module, just do it.

Questions? Suggestions? Find me at 1799824258@qq.com