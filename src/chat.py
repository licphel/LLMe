import mmg as mmg
import fetch as fetch
import load as load


def main():
    print("\n" + "=" * 50)
    print("[!] LLMe CLI")
    print("[!] '/help' for further info.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n[User] ").strip()

            if not user_input:
                continue

            handle_commands(user_input)
            if user_input.startswith("/"):
                continue

            print("[LLMe] ", end="", flush=True)
            mmg.chat(user_input)

        except KeyboardInterrupt:
            print("[!] LLMe chatting room exited.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


def handle_commands(user_input: str):
    # /help
    if user_input.startswith("/help"):
        print("[!] '/quit' to exit.")
        print("[!] '/switch <Model Name>' to switch between models.")
        print(
            "[!] '/load <Relative Path>' to scan all datasets in the given directory."
        )
        print("[!] '/clear' to clear cached data.")
        print("[!] '/train <Model Name>' to train a new model.")
        print(
            "[!] '/resume <Model Name> <Checkpoint Name> [Epochs]' to resume the training of a model."
        )
        print("[!] '/fetch hf <Name>' to download a dataset from huggingface.")

    # /quit
    if user_input.startswith("/quit"):
        print("[!] LLMe CLI exited.")
        exit(0)

    # /switch <name>
    if user_input.startswith("/switch "):
        datpath = user_input[8:].strip()

        if not datpath:
            print("[!] Please specify a model name. Usage: /switch <Model Name>")

        try:
            mmg.switch(datpath)
            print(f"[!] Successfully switched to {datpath}")
        except Exception as e:
            print(f"[!] Failed to switch model: {e}")

    # /train <name>
    if user_input.startswith("/train"):
        datpath = user_input[7:].strip()

        if not datpath:
            print("[!] Please specify a model name. Usage: /train <Model Name>")

        print(f"[!] Starting training for model: {datpath}")
        print("[!] This may take a while...")

        try:
            mmg.train(datpath)
            print(f"[!] Training completed! Model saved as: {datpath}")
        except Exception as e:
            print(f"[!] Training failed: {e}")

    # /resume <name> <fname> [epochs]
    if user_input.startswith("/resume"):
        parts = user_input.split()
        if len(parts) < 3:
            print("[!] Usage: /resume <Model Name> <Checkpoint Name> [additional_epochs]")

        model_name = parts[1]
        file_name = parts[2]
        additional_epochs: int = int(parts[3]) if len(parts) > 2 else 0

        print(f"[!] Resuming training for model: {model_name}")
        if additional_epochs:
            print(f"[!] Additional epochs: {additional_epochs}")
        print("[!] This may take a while...")
        print("[!] Press Ctrl+C to stop gracefully")

        try:
            stats = mmg.resume_train(model_name, file_name, additional_epochs)
            print(f"[!] Training resumed and completed!")
            print(f"    Model: {model_name}")
            print(f"    Total epochs: {stats['total_epochs']}")
            print(f"    Final loss: {stats['final_loss']:.4f}")
        except Exception as e:
            print(f"[!] Failed to resume training: {e}")

    # /clear
    if user_input.startswith("/clear"):
        confirm = input("[!] Are you sure you want to clear all cached data? (y/N): ")
        if confirm.lower() == "y":
            try:
                load.clear_data_cache()
                print("[!] Data cache cleared.")
            except Exception as e:
                print(f"[!] Failed to clear cache: {e}")
        else:
            print("[!] Operation cancelled.")

    # /load ?[relative_path]
    if user_input.startswith("/load"):
        datpath = user_input[6:].strip()

        if not datpath or datpath.isspace() or datpath == "":
            datpath = "data/"

        try:
            load.scan(datpath)
            print(f"[!] Successfully loaded {datpath}")
        except Exception as e:
            print(f"[!] Failed to load: {e}")

    # /fetch hf <name>
    if user_input.startswith("/fetch"):
        parts = user_input.split()
        if len(parts) < 3:
            print("[!] Usage: /fetch hf <dataset_name> [split]")

        if parts[1].lower() == "hf":
            dataset_name = parts[2]
            split = parts[3] if len(parts) > 3 else "all"

            print(f"[!] Fetching HuggingFace dataset: {dataset_name}")
            print("[!] This may take a while for first download...")

            try:
                stats = fetch.fetch_huggingface(dataset_name, split)
                print(f"[!] Dataset fetched successfully!")
                print(f"    Dataset: {dataset_name}")
                print(f"    Split: {split}")
                print(f"    Saved to: data/hf/{dataset_name.replace('/', '_')}")
                print(f"    Samples: {stats['samples']}")
                print(f"    Files: {stats['files']}")
            except Exception as e:
                print(f"[!] Failed to fetch dataset: {e}")
        else:
            print("[!] Only 'hf' is supported. Usage: /fetch hf <dataset_name>")


if __name__ == "__main__":
    main()
