import core


def main():
    print("\n" + "=" * 50)
    print("[!] LLMe chatting room")
    print("[!] Use '/quit' to exit.")
    print("[!] Use '/switch <Model Name>' to switch between models.")
    print("[!] Use '/tdat local' to load all local data from data/")
    print("[!] Use '/tdat hf <dataset>' to load data from HuggingFace")
    print("[!] Use '/tdat clear' to clear cached data")
    print("[!] Use '/tdat stats' to show data statistics")
    print("[!] Use '/train <Model Name>' to train a new model.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\n[User] ").strip()

            if user_input.lower() in ["/quit"]:
                print("[!] LLMe chatting room exited.")
                exit(0)
            if user_input.startswith("/switch "):
                model_name = user_input[8:].strip()

                if not model_name:
                    print(
                        "[!] Please specify a model name. Usage: /switch <Model Name>"
                    )
                    continue

                try:
                    core.switch(model_name)
                    print(f"[!] Successfully switched to {model_name}")
                except Exception as e:
                    print(f"[!] Failed to switch model: {e}")
                continue
            if user_input.startswith("/train "):
                model_name = user_input[7:].strip()

                if not model_name:
                    print("[!] Please specify a model name. Usage: /train <Model Name>")
                    continue

                print(f"[!] Starting training for model: {model_name}")
                print("[!] This may take a while...")

                try:
                    core.train(model_name)
                    print(f"[!] Training completed! Model saved as: {model_name}")
                except Exception as e:
                    print(f"[!] Training failed: {e}")
                continue

            if user_input.startswith("/tdat"):
                parts = user_input.split()
                if len(parts) < 2:
                    print("[!] Usage: /tdat [local|hf <name>|clear|stats]")
                    continue

                subcmd = parts[1].lower()
                if subcmd == "local":
                    print("[!] Loading local data from data/ directory...")
                    try:
                        stats = core.load_local_data()
                        print(f"[!] Local data loaded:")
                        print(f"    Total samples: {stats['total']}")
                        print(f"    By format: {stats['by_format']}")
                    except Exception as e:
                        print(f"[!] Failed to load local data: {e}")
                elif subcmd == "hf":
                    if len(parts) < 3:
                        print(
                            "[!] Please specify dataset name. Usage: /tdat hf <dataset>"
                        )
                        continue
                    dataset_name = parts[2]
                    split = parts[3] if len(parts) > 3 else "train"

                    print(
                        f"[!] Loading HuggingFace dataset: {dataset_name} (split: {split})..."
                    )
                    print("[!] This may take a while for first download...")

                    try:
                        stats = core.load_hf_data(dataset_name, split)
                        print(f"[!] HuggingFace data loaded:")
                        print(f"    Dataset: {dataset_name}")
                        print(f"    Split: {split}")
                        print(f"    Samples: {stats['samples']}")
                        print(f"    Text column: {stats.get('text_column', 'auto')}")
                    except Exception as e:
                        print(f"[!] Failed to load HuggingFace data: {e}")
                elif subcmd == "clear":
                    confirm = input(
                        "[!] Are you sure you want to clear all cached data? (y/N): "
                    )
                    if confirm.lower() == "y":
                        try:
                            core.clear_data_cache()
                            print("[!] Data cache cleared.")
                        except Exception as e:
                            print(f"[!] Failed to clear cache: {e}")
                    else:
                        print("[!] Operation cancelled.")
                elif subcmd == "stats":
                    try:
                        stats = core.get_data_stats()
                        if stats["total"] == 0:
                            print(
                                "[!] No data in cache. Use /tdat local or /tdat hf to load data."
                            )
                        else:
                            print("[!] Current data statistics:")
                            print(f"    Total samples: {stats['total']}")
                            print(f"    By source: {stats['by_source']}")
                            print(f"    By type: {stats['by_type']}")
                            if "sample_preview" in stats:
                                print(f"    Preview: {stats['sample_preview']}")
                    except Exception as e:
                        print(f"[!] Failed to get stats: {e}")

                else:
                    print(f"[!] Unknown subcommand: {subcmd}")
                    print("[!] Available: local, hf <name>, clear, stats")

                continue

            if not user_input:
                continue

            print("[LLMe] ", end="", flush=True)
            core.chat(user_input)

        except KeyboardInterrupt:
            print("[!] LLMe chatting room exited.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


if __name__ == "__main__":
    main()
