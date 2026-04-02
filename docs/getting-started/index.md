# Getting Started

## Installation

Clone the repo
```bash
git clone https://github.com/pvmodayil/agentic-pcb-design-assistant.git
cd agentic-pcb-design-assistant
```

Install dependencies using `uv`:

```bash
uv sync
```

### Configuration

Edit `src/agents/config/config.yaml` to configure agents, tools, and MCP servers.

### Setup

Create the tools as per **Tool Definition** and agents as per **Agent Definition**.
Call the Agents and orchestrator as per needed from the main.py file.

### Running

Run the main application:

```bash
uv run src/main.py
```