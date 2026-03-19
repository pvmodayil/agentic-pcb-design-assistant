from typing import Protocol, runtime_checkable

#---------------------------------------------------------
#                     Human Input
#---------------------------------------------------------
@runtime_checkable
class HumanInputProvider(Protocol):
    async def get_input(self, question: str) -> str: ...

# Standard Input method
class ConsoleInputProvider:
    async def get_input(self, question: str) -> str:
        try:
            return input(f"\n{question}\n> ")
        except Exception:
            return ""
        
class WebInputProvider:
    async def get_input(self, question: str) -> str:
        # TODO: Fill this when building UI
        return ""