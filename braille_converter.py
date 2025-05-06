import subprocess
import os

# Path to your Liblouis tables (update this path as needed)
os.environ["LOUIS_TABLEPATH"] = r"C:\\Users\\devin\\Downloads\\liblouis-3.33.0-win64\\share\\liblouis\\tables"

def check_table_exists(table):
    """Check if the Braille table exists at the specified path."""
    table_path = os.path.join(os.getenv("LOUIS_TABLEPATH", ""), table)
    return os.path.exists(table_path)

def convert_to_braille(text, table="en-us-g2.ctb", output_file="braille_output.txt"):
    """Convert text to Braille and save it to a file."""
    table_path = os.path.join(os.getenv("LOUIS_TABLEPATH", ""), table)
    
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Braille table {table} not found at {table_path}.")

    # Save input text to a temporary file
    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(text)

    # Command to run Liblouis translation with the correct table format
    lou_translate_path = r"C:\\Users\\devin\\Downloads\\liblouis-3.33.0-win64\\bin\\lou_translate"  # Full path to lou_translate executable
    command = [lou_translate_path, "--forward", table]  # Corrected command with --forward flag and table

    try:
        with open("input.txt", "r", encoding="utf-8") as input_file:
            result = subprocess.run(command, stdin=input_file, capture_output=True, text=True, check=True)
            # Write the Braille result to an output file
            with open(output_file, "w", encoding="utf-8") as braille_file:
                braille_file.write(result.stdout.strip())
        print(f"Braille output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during translation: {e.stderr}")
        raise Exception(f"Translation error: {e.stderr}")


