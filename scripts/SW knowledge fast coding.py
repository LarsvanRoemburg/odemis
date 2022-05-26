import numpy as np


def is_anagram(str1, str2):
    s1 = []
    s2 = []
    for i in range(len(str1)):
        s1.append(str1[i].lower())
    for j in range(len(str2)):
        s2.append(str2[j].lower())

    while " " in s1:
        s1.remove(" ")
    while " " in s2:
        s2.remove(" ")

    s1 = "".join(np.sort(s1))
    s2 = "".join(np.sort(s2))

    return s1 == s2


def sort_anagrams(list1):
    combined = np.zeros((len(list1), len(list1)))
    for i in range(len(list1)):
        for j in np.arange(i, len(list1)):
            if is_anagram(list1[i], list1[j]):
                combined[i, j] = i + 1

    end = np.zeros(len(list1))
    end2 = np.zeros(len(list1))
    combined[combined == 0] = np.inf
    for i in range(len(end)):
        end[i] = np.min(combined[:, i])
    un_end = np.unique(end)
    for i in range(len(un_end)):
        end2[end == un_end[i]] = i+1
    return end2


list1 = ['cat', 'dog', 'tac', 'god', 'act']

str1 = 'below'
str2 = 'elbow'

