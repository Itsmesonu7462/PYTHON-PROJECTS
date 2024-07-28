import os

# File to store the to-do list items
TODO_FILE = 'todos.txt'

def load_todos():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, 'r') as file:
        todos = file.readlines()
    return [todo.strip() for todo in todos]

def save_todos(todos):
    with open(TODO_FILE, 'w') as file:
        for todo in todos:
            file.write(f'{todo}\n')

def display_todos(todos):
    if not todos:
        print("No to-do items found.")
    else:
        print("\nTo-Do List:")
        for i, todo in enumerate(todos, start=1):
            print(f"{i}. {todo}")
    print()

def add_todo():
    todo = input("Enter a new to-do item: ").strip()
    if todo:
        todos.append(todo)
        save_todos(todos)
        print(f'Added: "{todo}"')

def delete_todo():
    display_todos(todos)
    try:
        index = int(input("Enter the number of the item to delete: ")) - 1
        if 0 <= index < len(todos):
            removed = todos.pop(index)
            save_todos(todos)
            print(f'Removed: "{removed}"')
        else:
            print("Invalid number.")
    except ValueError:
        print("Please enter a valid number.")

def main():
    global todos
    todos = load_todos()

    while True:
        print("\nOptions:")
        print("1. View To-Do List")
        print("2. Add To-Do Item")
        print("3. Delete To-Do Item")
        print("4. Exit")

        choice = input("Choose an option (1-4): ").strip()
        if choice == '1':
            display_todos(todos)
        elif choice == '2':
            add_todo()
        elif choice == '3':
            delete_todo()
        elif choice == '4':
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()
