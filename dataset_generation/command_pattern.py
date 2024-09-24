"""
This module defines a command pattern framework. It includes the abstract Command class
for defining specific commands and a CommandExecutor class to register and execute these commands.


Example usage:
```
if __name__ == "__main__":
    class MyCommand(Command):
        def execute(self, *args, **kwargs):
            print(f"Executing MyCommand with args: {args}, kwargs: {kwargs}")

        def add_arguments(self, parser):
            parser.add_argument('--myarg', type=int, help="An integer argument")

    executor = CommandExecutor()
    executor.register('mycommand', MyCommand())

    parser = argparse.ArgumentParser(description="Command pattern example")
    executor.add_commands_to_argparser(parser)
    args = parser.parse_args()
    executor.execute_from_args(args)
```
"""

import abc
import argparse
from typing import NoReturn


class AbstractCommand(abc.ABC):
    """
    Abstract base class for commands. Any specific command should inherit from this class and
    implement the execute method.
    """

    @abc.abstractmethod
    def execute(self, *args, **kwargs) -> NoReturn:
        """
        Execute the command with optional positional and keyword arguments.

        Args:
            *args: Positional arguments for the command.
            **kwargs: Keyword arguments for the command.
        """
        pass

    def help_str(self) -> str:
        """
        Returns a help string, suitable for use with an ArgumentParser.
        """
        return self.__class__.__doc__

    def add_arguments(self, parser: argparse.ArgumentParser) -> NoReturn:
        """
        Add command-specific arguments to the parser.

        Args: parser (argparse.ArgumentParser) The parser to add arguments to."""
        pass


class CommandExecutor:
    """
    CommandExecutor manages the registration and execution of commands.
    """

    def __init__(self, command_dict: dict[str, AbstractCommand] = {}):
        """
        Initialize the CommandExecutor with an optional dictionary of commands.

        Args:
            command_dict (dict): A dictionary where keys are command names and values are Command instances.
        """
        self.commands = command_dict

    def register(self, command_name: str, command: AbstractCommand):
        """
        Register a command with a specified name.

        Args:
            command_name (str): The name of the command to register.
            command (Command): The Command instance to register.
        """
        self.commands[command_name] = command

    def execute(self, command_name: str, *args, **kwargs):
        """
        Execute the command registered under the given name.

        Args:
            command_name (str): The name of the command to execute.

        Raises:
            ValueError: If the command name is not registered.
        """
        command = self.commands.get(command_name)
        if command:
            command.execute(*args, **kwargs)
        else:
            raise ValueError(
                f"Command '{command_name}' is not registered with this executor."
            )

    def execute_from_args(self, namespace: argparse.Namespace, *args, **kwargs):
        """
        Execute the first active command based on the argparse Namespace.

        Args:
            namespace (argparse.Namespace): Namespace containing command line arguments.
        """

        for cmd_name, is_active in [
            (cmd_key, cmd_key in namespace.__dict__.values())
            for cmd_key in self.get_registered_commands()
        ]:
            if is_active:
                self.execute(cmd_name, namespace)
                break

    def add_commands_to_argparser(self, parser: argparse.ArgumentParser):
        subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

        for name, command in self.commands.items():
            cmd_specific_parser = subparsers.add_parser(name, help=command.help_str())
            cmd_specific_parser.description = command.__class__.__doc__
            command.add_arguments(cmd_specific_parser)

        return parser

    def get_registered_commands(self):
        """
        Get a list of all registered command names.

        Returns:
            list: A list of registered command names.
        """
        return list(self.commands.keys())

    def __repr__(self):
        """
        Return a string representation of the CommandExecutor, listing all registered commands.

        Returns:
            str: A string representation of the CommandExecutor.
        """
        return f"Registered Commands: {self.get_registered_commands()}"


class SequentialCommandExecutor(CommandExecutor):
    def execute_from_args(self, namespace: argparse.Namespace, *args, **kwargs):
        """
        Execute the all active commands based on the argparse Namespace.

        Args:
            namespace (argparse.Namespace): Namespace containing command line arguments.
        """

        for cmd_name, is_active in [
            (c, namespace.__dict__[c]) for c in self.get_registered_commands()
        ]:
            if is_active:
                self.execute(cmd_name, namespace)

    def add_commands_to_argparser(self, parser: argparse.ArgumentParser):
        help_txt_suffix = "\nThis script has several modes: \n\n" + str(
            list(self.commands.keys())
        )

        parser.description += help_txt_suffix  # "\rn\t".join(help_txt_suffix)

        for name, command in self.commands.items():
            parser.add_argument(
                f"--{name}", action="store_true", help=command.help_str()
            )

        return parser
