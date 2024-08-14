if __name__ == '__main__':

    for data_name in ["ICEWS14", "ICEWS18", "ICEWS0515", "WIKI","YAGO"]:
        for file in ['valid']:
            output = file+"-1"
            infile = f"../data/{data_name}/{file}.txt"
            outfile = f"../data/{data_name}/{output}.txt"
            with open(infile, 'r') as file_a:
                lines = file_a.readlines()

            result_lines = []
            for line in lines:
                numbers = line.split()
                first_num = int(numbers[0]) - 1
                third_num = int(numbers[2]) - 1
                result_lines.append(f"{first_num} {numbers[1]} {third_num} {numbers[3]}")

            with open(outfile, 'w') as file_b:
                for result_line in result_lines:
                    file_b.write(result_line + '\n')
            print(f" the node of {data_name} {file} -1 successful ")