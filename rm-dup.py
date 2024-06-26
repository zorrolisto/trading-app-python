def remove_duplicate_versions(requirements_file):
    with open(requirements_file, 'r') as file:
        lines = file.readlines()

    packages = {}
    output_lines = []
    duplicates_removed = []

    for line in lines:
        if line.strip():  # Ignore empty lines
            parts = line.strip().split('==')
            package_name = parts[0].lower()

            if package_name in packages:
                # Check if the current version is newer than the stored version
                current_version = parts[1] if len(parts) > 1 else None
                stored_version = packages[package_name]

                if current_version and stored_version and current_version > stored_version:
                    duplicates_removed.append(f"Removed duplicate: {package_name}=={stored_version}")
                    packages[package_name] = current_version
                    output_lines.append(line.strip() + '\n')
            else:
                packages[package_name] = parts[1] if len(parts) > 1 else None
                output_lines.append(line.strip() + '\n')

    with open(requirements_file, 'w') as file:
        file.writelines(output_lines)

    print(f"Duplicates removed and saved to {requirements_file}")
    if duplicates_removed:
        print("Duplicates removed:")
        for duplicate in duplicates_removed:
            print(duplicate)

# Ejemplo de uso:
if __name__ == "__main__":
    requirements_file = "requirements.txt"
    remove_duplicate_versions(requirements_file)

