"""Day 1 smoke test: chat with the agent from the CLI, see every step."""

from harness.agent import Agent


def main() -> None:
    agent = Agent()
    print("Athena (Day 1 harness). Ctrl-C to quit.\n")
    history: list[dict] = []
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue

        result = agent.run(user_input, history=history)

        for step in result.trace:
            if step.kind == "tool_call":
                print(f"  [tool_call] {step.tool_name}({step.content})")
            elif step.kind == "tool_result":
                print(f"  [tool_result] {step.tool_name} -> {step.content}")
            elif step.kind == "thought":
                print(f"  [thought] {step.content}")

        print(f"athena> {result.answer}\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": result.answer})


if __name__ == "__main__":
    main()
