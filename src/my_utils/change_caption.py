


if __name__ == '__main__':
    data = []
    with open(
            '/data/lanyunwei/my_experiment_results/capation_list/caption_test.txt',
            "r") as f:
        lines = f.readlines()

        for line in lines:
            # 根据条件修改
            if line.find('smoggy') != -1 :
                new_line = line.replace("smoggy", '')
            elif line.find('smog') != -1:
                new_line = line.replace("smog", '')
            elif line.find('hazy') != -1:
                new_line = line.replace("hazy", '')
            elif line.find('haze') != -1:
                new_line = line.replace("haze", '')
            elif line.find('foggy') != -1:
                new_line = line.replace("foggy", '')
            elif line.find('fog') != -1:
                new_line = line.replace("fog", '')

            else:
                new_line = line
            data.append(new_line)
    with open(
            '/data/lanyunwei/my_experiment_results/caption_list_dehaze/caption_test.txt',
            "w") as f:
        for i in data:
            f.writelines(i)




