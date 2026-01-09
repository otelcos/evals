---
id: agents
title: Telco Specific Agents
sidebar_label: Telco Specific Agents
sidebar_position: 2
---

# Telco Specific Agents

Open Telco provides a library of specialized agents designed for telecommunications tasks and workflows.

## Overview

Our telco-specific agents are built on top of language models and enhanced with:

- Domain-specific knowledge
- Tool use capabilities
- Network environment integration
- Telecom protocol understanding

## Available Agents

### Network Diagnostics Agent

An agent specialized in analyzing network logs and identifying root causes of issues.

**Capabilities:**
- Parse and analyze network logs
- Identify patterns indicating problems
- Suggest remediation steps
- Generate diagnostic reports

**Usage:**
```python
from open_telco.agents import NetworkDiagnosticsAgent

agent = NetworkDiagnosticsAgent(model="gpt-4o")
result = agent.diagnose(logs=network_logs)
```

### Configuration Agent

An agent for generating and validating network configurations.

**Capabilities:**
- Generate YAML configurations for 5G networks
- Validate configurations against standards
- Suggest optimizations
- Handle AMF, UPF, and slice configurations

### Standards Expert Agent

An agent with deep knowledge of 3GPP and other telecom standards.

**Capabilities:**
- Answer questions about standards
- Find relevant specification sections
- Explain technical concepts
- Cross-reference between releases

## Agent Architecture

```
┌─────────────────────────────────────┐
│           Language Model            │
├─────────────────────────────────────┤
│         Domain Knowledge            │
│    (Telecom standards, protocols)   │
├─────────────────────────────────────┤
│              Tools                  │
│  (bash, python, network simulators) │
├─────────────────────────────────────┤
│        Network Environment          │
│     (Docker/Kathara sandboxes)      │
└─────────────────────────────────────┘
```

## Building Custom Agents

### Basic Agent Structure

```python
from inspect_ai import Task, task
from inspect_ai.solver import generate, use_tools, system_message
from inspect_ai.tool import bash, python

SYSTEM_PROMPT = """
You are a telecommunications expert agent specialized in...
"""

@task
def custom_agent() -> Task:
    return Task(
        dataset=load_tasks(),
        solver=[
            system_message(SYSTEM_PROMPT),
            use_tools([bash(), python()]),
            generate(),
        ],
        scorer=evaluate_agent(),
        sandbox="docker",
    )
```

### Adding Domain Knowledge

Enhance agents with telecom-specific knowledge:

```python
from inspect_ai.solver import system_message

TELECOM_KNOWLEDGE = """
Key 5G Network Functions:
- AMF (Access and Mobility Management Function)
- SMF (Session Management Function)
- UPF (User Plane Function)
- NRF (Network Repository Function)
...
"""

solver = [
    system_message(TELECOM_KNOWLEDGE),
    # ... rest of solver chain
]
```

## Agent Evaluation

We evaluate agents on:

| Metric | Description |
|--------|-------------|
| **Task Completion** | Successfully completing assigned tasks |
| **Accuracy** | Correctness of generated outputs |
| **Efficiency** | Number of steps to completion |
| **Safety** | Avoiding harmful operations |

## Best Practices

1. **Start Simple**: Begin with basic tool use before adding complexity
2. **Validate Outputs**: Always verify agent-generated configurations
3. **Use Sandboxes**: Run agents in isolated environments
4. **Monitor Behavior**: Log agent actions for debugging
5. **Iterate on Prompts**: Refine system prompts based on performance

## Contributing Agents

We welcome contributions of new agents:

1. Follow the [contribution guidelines](/docs/contributing)
2. Include comprehensive documentation
3. Provide evaluation benchmarks
4. Ensure safety considerations are addressed

## Roadmap

Upcoming agent capabilities:

- Real-time network monitoring integration
- Multi-agent collaboration
- Enhanced reasoning chains
- Broader protocol support
