import os
import json
import subprocess
import importlib.metadata
from packaging import version
import sys


def save_json(my_data, file_name):
    if not os.path.exists('data'):
        os.makedirs('data')

    file_name = file_name.replace('.json', '') + '.json'
    path = os.path.join('data', file_name)

    with open(path, 'w') as f:
        json.dump(my_data, f, ensure_ascii=True, indent=4)


def install_requirements(requirements: list[str]) -> None:
    """
    Installs or upgrades packages based on the version requirements specified in the list.

    :param requirements: List of packages with potential version specifiers.
    """
    for requirement in requirements:
        package_info = requirement.split('==') if '==' in requirement else (
            requirement.split('>=') if '>=' in requirement else (
                requirement.split('<=') if '<=' in requirement else [requirement]))
        package_name = package_info[0]
        required_version_spec = requirement[len(package_name):]

        try:
            # Check if the package is already installed
            dist = importlib.metadata.distribution(package_name)
            installed_version = version.parse(dist.version)
            if required_version_spec:
                # Extract the operator and the version from requirement
                operator = required_version_spec[:2] if required_version_spec[1] in ['=', '>'] else \
                    required_version_spec[0]
                required_version = version.parse(required_version_spec.lstrip(operator))

                # Version comparison based on the operator
                if ((operator == '==' and installed_version == required_version) or
                        (operator == '>=' and installed_version >= required_version) or
                        (operator == '<=' and installed_version <= required_version)):
                    print(f"Package {package_name} already installed and meets the requirement {requirement}.")
                else:
                    print(
                        f"Package {package_name} version {installed_version} does not meet the requirement {requirement}, upgrading...")
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", f"{package_name}{required_version_spec}"])
            else:
                print(f"Package {package_name} is already installed.")
        except importlib.metadata.PackageNotFoundError:
            # Package is not installed, install it with the specified version
            print(f"Package {package_name} is not installed, installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error installing or upgrading package {package_name}: {e}")
