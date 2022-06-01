import numpy as np


# def is_anagram(str1, str2):
#     s1 = []
#     s2 = []
#     for i in range(len(str1)):
#         s1.append(str1[i].lower())
#     for j in range(len(str2)):
#         s2.append(str2[j].lower())
#
#     while " " in s1:
#         s1.remove(" ")
#     while " " in s2:
#         s2.remove(" ")
#
#     s1 = "".join(np.sort(s1))
#     s2 = "".join(np.sort(s2))
#
#     return s1 == s2
#
#
# def sort_anagrams(list1):
#     combined = np.zeros((len(list1), len(list1)))
#     for i in range(len(list1)):
#         for j in np.arange(i, len(list1)):
#             if is_anagram(list1[i], list1[j]):
#                 combined[i, j] = i + 1
#
#     end = np.zeros(len(list1))
#     end2 = np.zeros(len(list1))
#     combined[combined == 0] = np.inf
#     for i in range(len(end)):
#         end[i] = np.min(combined[:, i])
#     un_end = np.unique(end)
#     for i in range(len(un_end)):
#         end2[end == un_end[i]] = i+1
#     return end2
#
#
# list1 = ['cat', 'dog', 'tac', 'god', 'act']
#
# str1 = 'below'
# str2 = 'elbow'


print("hi")

def fizzbuzz():
    for i in range(1, 101):
        if i % 3 == 0 and i % 5 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)


# fizzbuzz()


def max_profit(prices_list):
    if prices_list[0] < prices_list[1]:
        local_minima = [0]
    else:
        local_minima = []

    local_maxima = []
    for i in range(1, len(prices_list)-1):
        if prices_list[i] < prices_list[i+1] and prices_list[i] <= prices_list[i-1]:
            local_minima.append(i)
        elif prices_list[i] > prices_list[i+1] and prices_list[i] >= prices_list[i-1]:
            local_maxima.append(i)
    if prices_list[-1] >= prices_list[-2]:
        local_maxima.append(len(prices_list)-1)
    local_maxima = np.array(local_maxima)
    local_minima = np.array(local_minima)
    minima_not_used = np.ones(len(local_minima), dtype=bool)
    maxima_not_used = np.ones(len(local_maxima), dtype=bool)

    for i in range(len(local_minima)):
        for j in range(len(local_maxima)):
            if minima_not_used[i] and maxima_not_used[j]:
                if local_minima[i] < local_maxima[j]:
                    print("Buy at day {} ({})".format(local_minima[i]+1, prices_list[local_minima[i]]))
                    print("Sell at day {} ({})".format(local_maxima[i]+1, prices_list[local_maxima[i]]))
                    minima_not_used[i] = False
                    maxima_not_used[j] = False


market = np.random.rand(100)*np.linspace(0.5, 1.5, 100)

max_profit(market)
