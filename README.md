# Agentic PCB Design Assistant

An intelligent, agent-based PCB (Printed Circuit Board) design automation assistant powered.

## Overview

This project implements a sophisticated agentic framework for automating PCB design workflows. It uses **pydantic-ai** as the core agent framework, integrates with external tools via MCP servers (both local and remote), and provides a checkpoint-based workflow management system to track progress through complex design tasks. The core architecture is based on a deterministic workflow with fine-grained control.

## Architecture

### Core Components

- **PCB Agent** (`src/agents/pcb_agent.py`): Abstract base class for domain-specific agents with built-in checkpoint tracking, state management, and tool integration
- **Tool Registry** (`src/agents/tool_registry.py`): Centralized registry for managing and invoking agent tools
- **MCP Tool Adapter** (`src/agents/mcp_server_builder.py`): Wrapper for integrating external MCP (Model Context Protocol) servers as tools
- **Memory Manager** (`src/agents/memory_manager.py`): Manages agent memory, context summarization, and conversation history
- **Data Models** (`src/agents/data_models.py`): Core Pydantic models for agents, workflows, checkpoints, and results
- **LLM Model** (`src/agents/llm_model.py`): Integration with language models for agent reasoning
- **Configuration** (`src/agents/config/`): Settings management with YAML-based configuration

### Key Features

- **Checkpoint-Based Workflows**: Track progress through multi-step design processes with checkpoint validation
- **MCP Integration**: Support for both local (stdio) and remote (SSE/HTTP) MCP servers
- **Tool Management**: Dynamic tool registration and invocation with type-safe definitions
- **Agent State Management**: Full state tracking including pending checkpoints, actions, and results
- **Memory & Summarization**: Built-in memory management with context summarization for extended workflows
- **Error Handling**: Robust error handling with execution timing and detailed error messages

## Technology Stack

- **Framework**: pydantic-ai >= 1.41.0 (Agent framework and LLM integration)
- **Data Validation**: Pydantic >= 2.12.0
- **Python**: >= 3.14

## Project Structure

```
src/
├── agents/              # Core agent implementation
│   ├── pcb_agent.py    # Base agent class
│   ├── tool_registry.py # Tool management
│   ├── mcp_server_builder.py # MCP integration
│   ├── memory_manager.py # Memory management
│   ├── llm_model.py    # LLM configuration
│   ├── data_models.py  # Pydantic models
│   └── config/         # Configuration files
├── infrastructure/     # Infrastructure utilities
├── orchestrator/       # Workflow orchestration
├── protocols/          # Protocol definitions
├── tools/              # Tool implementations
└── utils/              # Utility functions
```

## Getting Started

### Installation

Install dependencies using `uv`:

```bash
uv sync
```

### Configuration

Edit `src/agents/config/config.yaml` to configure agents and MCP servers.

### Running

Run the main application:

```bash
uv run src/main.py
```

## Contributing

This project is part of the [Information Processing Lab's](https://dt.etit.tu-dortmund.de/en/) agentic AI research initiative.
