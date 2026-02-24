import core

def main():
    print("\n" + "="*50)
    print("[!] LLMe chatting room")
    print("[!] Use '/quit' to exit.")
    print("[!] Use '/switch <Model Name>' to switch between models.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n[User] ").strip()

            if user_input.lower() in ['/quit']:
                print("[!] LLMe chatting room exited.")
                break
            if user_input.startswith('/switch '):
                model_name = user_input[8:].strip()
                
                if not model_name:
                    print("[!] Please specify a model name. Usage: /switch <Model Name>")
                    continue
                
                try:
                    core.switch(model_name)
                    print(f"[!] Successfully switched to {model_name}")
                except Exception as e:
                    print(f"[!] Failed to switch model: {e}")
                continue

            if not user_input:
                continue

            print("[LLMe] ", end='', flush=True)
            core.chat(user_input)
            
        except KeyboardInterrupt:
            print("[!] LLMe chatting room exited.")
            break
        except Exception as e:
            print(f"[!] Error: {e}")

if __name__ == "__main__":
    main()