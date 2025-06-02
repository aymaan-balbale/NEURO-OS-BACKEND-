# main.py

import time
from dispatcher import Dispatcher

def main():
    """
    Main function to run the interactive AI OS command-line interface.
    """
    dispatcher = Dispatcher()
    print("\n-----------------------------------------------------")
    print("Welcome to your Local AI OS!")
    print("Type your queries, or 'exit' to quit.")
    print("-----------------------------------------------------")

    while True:
        try:
            user_query = input(f"[{dispatcher.llama3_agent.current_working_directory}]$ Your AI: ")
            
            if user_query.lower() == 'exit':
                print("Exiting AI OS. Goodbye!")
                break
            
            if not user_query.strip():
                continue

            response = dispatcher.dispatch_request(user_query)
            print(f"\nAI Response:\n{response}\n")

        except KeyboardInterrupt:
            print("\nExiting AI OS. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred during AI processing: {e}")
            print("Please try again or check the logs above for more details.")
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
