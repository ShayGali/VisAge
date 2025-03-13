import csv
import os
import random

def get_age_range(age:int):
    if 0 <= age <= 3:
        return '(0, 2)'
    elif 4 <= age <= 7:
        return '(4, 6)'
    elif 8 <= age <= 14:
        return '(8, 12)'
    elif 15 <= age <= 23:
        return '(15, 20)'
    elif 24 <= age <= 35:
        return '(25, 32)'
    elif 36 <= age <= 45:
        return '(38, 43)'
    elif 46 <= age <= 59:
        return '(48, 53)'
    elif 60 <= age:
        return '(60, 100)'
    else:
        raise ValueError(f'Age not in any range: {age}')
    
def get_data_from_img_name(img_name: str):
    """
    the image foramt is: [age]_[gender]_[race]_[date&time].jpg
    [age] is an integer from 0 to 116, indicating the age
    [gender] is either 0 (male) or 1 (female)
    [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
    
    we will return the age range, gender (f/m), race (as number)
    :param img_name: 
    :return: tuple (age, gender, race) 
    """

    data_split = img_name.split('_')
    if len(data_split) != 4:
        return None, None, None
    age = int(data_split[0])
    gender = data_split[1]
    race = int(data_split[2])

    # get age range
    age_range = get_age_range(age)
    if gender == '0':
        gender = 'm'
    else:
        gender = 'f'
    
    return age_range, gender, race


def create_csv_file():
    """
    create a csv file with the following columns: 
    age, gender, race, img

    we want ~50 from each (age_range, gender, race) combination
    the data is in: data/UTKFace/
    """
    # delete the file if it exists
    if os.path.exists('data/UTKFace.csv'):
        os.remove('data/UTKFace.csv')


    # define the number of samples we want from each category
    num_samples = 50

    # define memory to store the number of samples we have
    samples = {} # key: (age_range, gender, race), value: number of samples

    # create the csv file
    with open('data/UTKFace.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # go thorugh each file, check if it is an image, and saev the data
        files = os.listdir('data/UTKFace/')
        # shuffle the files
        random.shuffle(files)


        for file in files:
            if file.endswith('.jpg'):
                age_range, gender, race = get_data_from_img_name(file)
                if age_range is None:
                    continue

                # check if we have enough samples
                key = f"{age_range}_{gender}_{race}"
                if key not in samples:
                    samples[key] = 0
                if samples[key] >= num_samples:
                    continue
                samples[key] += 1

                writer.writerow([age_range, gender, race, file])

    return samples # return the number of samples we have

#function that put all of the images that in the csv file in a folder
def create_folder():
    with open('data/UTKFace.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            img_path = os.path.join('data/UTKFace/', row[3])
            new_path = os.path.join('data/UTKFace2/', row[3])
            os.rename(img_path, new_path)
            print(f'{img_path} -> {new_path}')
    
    


if __name__ == '__main__':
    samples = create_csv_file()
    for key, value in samples.items():
        print(f'{key}: {value}')
    
    # count the number of samples
    total = 0
    for key, value in samples.items():
        total += value
    print(f'Total samples: {total}')

    create_folder()
        



