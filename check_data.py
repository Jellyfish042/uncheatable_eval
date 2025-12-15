import json
import os
import sys
import random


class JSONLBrowser:
    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.current_index = 0
        self.load_data()

        print(f"Loaded {len(self.data)} JSON objects from {self.filename}")

    def load_data(self):
        """Load the JSONL file line by line or load JSON file as a list"""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                if self.filename.lower().endswith(".jsonl"):
                    for line in f:
                        try:
                            json_obj = json.loads(line.strip())
                            self.data.append(json_obj)
                        except json.JSONDecodeError:
                            print(f"Warning: Skipped invalid JSON line: {line}")
                else:  # Assume it's a regular JSON file
                    try:
                        json_content = json.load(f)
                        if isinstance(json_content, list):
                            self.data = json_content
                        else:
                            print("Error: JSON file content is not a list")
                            sys.exit(1)
                    except json.JSONDecodeError:
                        print(f"Error: Invalid JSON in file {self.filename}")
                        sys.exit(1)

            print(f"Loaded {len(self.data)} JSON objects from {self.filename}")
        except FileNotFoundError:
            print(f"Error: File '{self.filename}' not found")
            sys.exit(1)

    def display_current(self):
        """Display the current JSON object"""
        os.system("cls" if os.name == "nt" else "clear")  # Clear the screen
        if not self.data:
            print("No data to display")
            return

        print(f"Object {self.current_index + 1}/{len(self.data)}")
        print("-" * 40)
        try:
            current_obj = self.data[self.current_index]
            if "content" in current_obj:
                other_fields = {k: v for k, v in current_obj.items() if k != "content"}
                print(json.dumps(other_fields, indent=2, ensure_ascii=False))
                content = current_obj["content"]
                print(content)
            else:
                formatted_json = json.dumps(current_obj, indent=2, ensure_ascii=False)
                print(formatted_json)
        except IndexError:
            print("Error: Invalid index")
        print("-" * 40)
        print("Controls: [n]ext | [p]revious | [f]irst | [l]ast | [g]oto | [s]earch | [r]andom | [q]uit")

    def next_item(self):
        """Move to the next JSON object"""
        if self.data and self.current_index < len(self.data) - 1:
            self.current_index += 1

    def prev_item(self):
        """Move to the previous JSON object"""
        if self.data and self.current_index > 0:
            self.current_index -= 1

    def first_item(self):
        """Move to the first JSON object"""
        if self.data:
            self.current_index = 0

    def last_item(self):
        """Move to the last JSON object"""
        if self.data:
            self.current_index = len(self.data) - 1

    def random_item(self):
        """Move to a random JSON object"""
        if self.data:
            self.current_index = random.randint(0, len(self.data) - 1)

    def goto_item(self):
        """Go to a specific JSON object by index"""
        if not self.data:
            return

        try:
            index = int(input("Enter index (1-{}): ".format(len(self.data)))) - 1
            if 0 <= index < len(self.data):
                self.current_index = index
            else:
                input("Invalid index. Press Enter to continue...")
        except ValueError:
            input("Invalid input. Press Enter to continue...")

    def search_items(self):
        """Search for a key or value in the JSON objects"""
        if not self.data:
            return

        search_term = input("Enter search term: ").lower()
        if not search_term:
            return

        found_indices = []
        for i, item in enumerate(self.data):
            json_str = json.dumps(item, ensure_ascii=False).lower()
            if search_term in json_str:
                found_indices.append(i)

        if found_indices:
            print(f"Found {len(found_indices)} matches")
            for i, idx in enumerate(found_indices):
                print(f"{i+1}. Object {idx+1}")

            try:
                choice = int(input("Select a match (1-{}) or 0 to cancel: ".format(len(found_indices))))
                if 1 <= choice <= len(found_indices):
                    self.current_index = found_indices[choice - 1]
            except ValueError:
                pass
        else:
            input("No matches found. Press Enter to continue...")

    def run(self):
        """Main loop for the JSONL browser"""
        if not self.data:
            print("No data to browse")
            return

        while True:
            self.display_current()
            command = input("Command: ").lower()

            if command == "n":
                self.next_item()
            elif command == "p":
                self.prev_item()
            elif command == "f":
                self.first_item()
            elif command == "l":
                self.last_item()
            elif command == "g":
                self.goto_item()
            elif command == "s":
                self.search_items()
            elif command == "r":
                self.random_item()
            elif command == "q":
                break


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python jsonl_browser.py <jsonl_file>")
        sys.exit(1)

    browser = JSONLBrowser(sys.argv[1])
    browser.run()
