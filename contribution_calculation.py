import pandas as pd
import numpy as np


class ElectricBillCalculator:
    def __init__(self, split_list, bill_per_day=10.0):
        self.split = split_list
        self.cost = bill_per_day

    def calculate(self):
        sorted_split = sorted(self.split)
        name_list = []
        days_list = []
        counter = 0
        total_people = len(sorted_split)
        contribution_array = np.zeros(total_people)

        for idx, (days, name) in enumerate(sorted_split):
            add_array = np.zeros(total_people)
            share = (days - counter) * (self.cost / (total_people - idx))
            add_array[idx:] = share
            name_list.append(name)
            days_list.append(days)
            contribution_array += add_array
            counter = days

        contribution = pd.DataFrame({'Name': name_list, 'Share': contribution_array, "Days Present": days_list})

        return contribution


if __name__ == '__main__':
    ebc = ElectricBillCalculator(split_list=[(20, 'Aswin'), (15, 'Shriram')], bill_per_day=12.0)
    df = ebc.calculate()

    print(df)
