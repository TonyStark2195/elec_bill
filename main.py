from contribution_calculation import ElectricBillCalculator
from streamlit_interface import UI


def main():
    ui = UI()
    present_list, cost_per_day = ui.build_interface()

    ebc = ElectricBillCalculator(split_list=present_list, bill_per_day=cost_per_day)
    df = ebc.calculate()

    ui.display(df)


if __name__ == '__main__':
    main()
