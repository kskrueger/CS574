import csv
import numpy as np
import datetime


def read_csv(path):
    with open(path, 'r') as f:
        csv_file = csv.reader(f)
        lines = [line for line in csv_file]
    return lines


order_dataset = read_csv("dataset/Groceries_dataset.csv")[1:]
# grocery_dataset = read_csv("dataset/groceries - groceries.csv")[1:]

orders_dict = {}
for order in order_dataset:
    order_num, time, item = order
    time_split = time.split("-")
    time = "-".join(reversed(time_split))
    if order_num in orders_dict:
        if time in orders_dict[order_num]:
            orders_dict[order_num][time].append(item)
        else:
            orders_dict[order_num][time] = [item]
    else:
        orders_dict[order_num] = {}
        orders_dict[order_num][time] = [item]


daily_dict = {}
for order in order_dataset:
    order_num, time, item = order
    time_split = time.split("-")
    time = "-".join(reversed(time_split))
    if time in daily_dict:
        if order_num in daily_dict[time]:
            daily_dict[time][order_num].append(item)
        else:
            daily_dict[time][order_num] = [item]
    else:
        daily_dict[time] = {}
        daily_dict[time][order_num] = [item]

daily_item_dict = {}
foods_list = set()
for order in order_dataset:
    order_num, time, item = order
    time_split = time.split("-")
    time = "-".join(reversed(time_split))
    if time in daily_item_dict:
        if order_num in daily_item_dict[time]:
            if item in daily_item_dict[time][order_num]:
                daily_item_dict[time][order_num][item] += 1
            else:
                daily_item_dict[time][order_num][item] = 1
        else:
            daily_item_dict[time][order_num] = {}
            daily_item_dict[time][order_num][item] = 1
    else:
        daily_item_dict[time] = {}
        daily_item_dict[time][order_num] = {}
        daily_item_dict[time][order_num][item] = 1
    foods_list.add(item)


food_encodings = dict([(k, v) for v, k in enumerate(foods_list)])
days_list = sorted(daily_item_dict.keys())
day_encodings = dict([(k, v) for v, k in enumerate(days_list)])
day_of_week = np.array([datetime.datetime(*[int(v) for v in ymd.split('-')]).weekday() for ymd in days_list])
# day_total_sales = np.vstack([sum([len(o) for o in vals.values()]) for key, vals in daily_dict.items()])

day_item_sales = np.zeros((len(daily_dict.keys()), len(food_encodings)), dtype=int)
for k, v in daily_item_dict.items():
    for order in v.values():
        for i, val in order.items():
            day_item_sales[day_encodings[k]][food_encodings[i]] += val

print(max(orders_dict.values(), key=lambda x: len(x)))
print(max(daily_dict.values(), key=lambda x: len(x)))

