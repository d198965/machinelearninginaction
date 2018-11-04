#!/usr/bin/python
# -*- coding: UTF-8 -*-

class TreeNode:
    def __init__(self, nameValue, occurNumber, parentNode):
        self.name = nameValue
        self.parent = parentNode
        self.count = occurNumber
        self.linkNode = None
        self.children = {}

    def inc(self, numberOccur):
        self.count = self.count + numberOccur

    def disp(self, ind=1):
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)
