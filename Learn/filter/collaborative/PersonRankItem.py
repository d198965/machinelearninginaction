#!/usr/bin/python
# -*- coding: UTF-8 -*-
class PersonRankItem:
    def __init__(self, item, relationIds):
        self.item = item
        self.relationIds = relationIds
        self.value = 0
        self.temValue = 0

    def addSum(self):
        self.value = self.temValue
        self.temValue = 0

    def __eq__(self, other):
        if isinstance(other, PersonRankItem):
            return self.item == other.item
        else:
            return False

    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.item)

    def __cmp__(self, other):
        if self.value > other.value:
            return -1
        elif self.value < other.value:
            return 1
        else:
            return 0
