def text_opcodes_to_binary(input_file, output_file):
    # Read the text file
    with open(input_file, "r") as f:
        text_opcodes = f.read().strip()

    text_opcodes = ''.join(text_opcodes.split())

    # Convert text opcodes to binary
    binary_data = bytearray()
    for i in range(0, len(text_opcodes), 2):
        # Take two characters at a time
        byte = text_opcodes[i:i+2]
        # Convert the two-character string to an integer and then to a byte
        byte_value = int(byte, 16)
        binary_data.append(byte_value)

    with open(output_file, "wb") as f:
        f.write(binary_data)

def test_generate_matrix_binary():
    # pow hash 1f9c3313aed61df15e978d845514ae8f3992d7aeb56c6e2e79df6b23028cc981
    text_opcodes_to_binary(r"C:\Users\idofi\AppData\Roaming\JetBrains\RustRover2024.1\scratches\scratch_27.txt", r"C:\Users\idofi\downloads\asdasd.bin")