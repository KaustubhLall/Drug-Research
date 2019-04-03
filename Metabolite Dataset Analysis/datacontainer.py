### File - datacontainer.py
import csv
from copy import deepcopy

'''
This file defines a data container object. The purpose is to be able to extract certain columns from the dataset by
passing in a list of indices to the column. The datastructure supports the following operations:


1. Visualization: You can print the entire dataset by simply passing the object into a print function.
It should generate a table.

2. Initialization: To initialize a DataContainer, simply call it with a single argument - the csv file that contains your data.
The DataContainer will do automatic tagging of float/int data types with categorical data, however it is not designed to 
Do a mix of both. Having entries like "4-fold" etc may throw off the auto type inference, and is highly discouraged
because is it not a guarantee that error checking will catch such discrepancies.
Ex dc = DataContainer('myfile.csv')

3. Removing Columns: Once a data container is made, columns MAY NOT be removed. This feature is implemented by the 
self.dropcol method, However, the implementation is buggy. It will permanently throw an assertion error.

4. Fetching Columns: Columns may, however be fetched using the getcol(index) method. 
Doing so will return a (column header, column contents) tuple, the latter of which is simply a list.
To get a column by its header, simply do dc.getcol(dc.header.index('name of column header')).

5. File I/O : You can write the contents of a DataContainer object to file using the dc.writeCSV('destinationfile.csv')
method. 

6. Subsetting : If you want to get a copy of the data container with only some (a subset of its) columns, you can use
the subsetDataContainer(source, cols), which returns a copy of the source DataContainer with only the columns specified 
by the parameter cols =[indices of cols].



'''


def subsetDataContainer(source, cols):
    '''
    Takes columns from an existing Data Container object and makes a new one with the new cols.
    :param source: the source container to pull cols from.
    :param cols: the cols we want to pull.
    :return: new data container with exactly the cols you want.
    '''
    cols = sorted(cols, reverse=True)
    newcontainer = DataContainer(False)
    newcontainer.numcols = len(cols)
    newcontainer.numrows = source.numrows
    newcontainer.categoricalMap = {}

    for i in range(len(cols)):
        c = cols[i]
        newcontainer.header.append(source.header[c])
        newcontainer.dtypes.append(source.dtypes[c])
        if c in source.categoricalMap:
            newcontainer.categoricalMap[i] = source.categoricalMap[c]

    # add rows to datamatrix
    newcontainer.dataMatrix = []

    for row in source.dataMatrix:
        newrow = [row[i] for i in cols]
        newcontainer.dataMatrix.append(newrow)

    return newcontainer


def splitDataset(splits=[80, 10, 10], seed=1):
    '''
    Parses a csv file, creates a training, test and validation set. Ensures training set is balanced in classes.
    Once we have our standard test training and validation splits, we will use the files to create one data container
    for each, and then subset all possible combinations of features.

    This feature is a TODO, will not necessarily be implemented.

    :param splits: splits to use for training, test and validation.
    :param seed: numpy seed to use for shuffling the dataset.
    :return: None. Writes to disk.
    '''
    pass


class DataContainer():
    # header = []
    # numrows = 0
    # numcols = 0
    # dataMatrix = []
    # dtypes = []
    # categoricalMap = {}

    def __repr__(self):
        '''
        Overloaded print method.
        :return: string containing object info.
        '''
        cellwidth = 15
        s = ''
        s += '|' + '|'.join([('%s' % x).center(cellwidth) for x in self.header]) + '|\n'
        s += '=' * len(s) + '\n'

        for i in range(len(self.dataMatrix)):
            row = self.dataMatrix[i]
            s += '|' + '|'.join([('%s' % x).center(cellwidth) for x in row]) + '|\n'

        # :-1 ensures last newline is not printed
        return s[:-1]

    def __init__(self, fname, bannedcols=[]):
        '''
        Initializes the datacontainer from a given csv.
        :param fname: filename to initialize from.
        :param bannedcols: index of columns to drop. TODO this feature is not currently supported.
        :return: the initialized datacontainer.
        '''

        self.header = []
        self.numrows = 0
        self.numcols = 0
        self.dataMatrix = []
        self.dtypes = []
        self.categoricalMap = {}

        # this is a special case where we manually create a new data container
        if fname == False:
            return

        with open(fname, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for i, row in enumerate(reader):
                if len(self.header) == 0:
                    self.header = row
                else:
                    # here we need to find the datatype of each elem
                    # also need to automatically parse the datatype for subsequent rows without loss of generality
                    # need a method to handle categorical data as well as a to-and forth conversion
                    # one way to do this is by post-processing entire columns
                    self.dataMatrix.append(row)

        self.dtypes = ['unknown' for x in self.dataMatrix[0]]


        assert len(self.header) == len(self.dataMatrix[0])
        # update self parameters
        self.numcols = len(self.dataMatrix[0])
        self.numrows = len(self.dataMatrix)

        # post-process the columns to get their dtype, etc.
        for i in range(self.numcols):
            self.mapcol(i)
        # clean up the cols to drop
        for e in sorted(bannedcols, reverse=True):
            self.dropcol(e)

    def dropcol(self, col):
        '''
        Drops a column from the data matrix.
        :param col: index of column to drop.
        :return: None.
        '''
        # NOTE CURRENTLY A BUG IN THIS METHOD - DO NOT USE OR PROGRAM WILL NOT TERMINATE, see
        # @subsetDataContainer method
        assert False
        assert self.numcols >= 1
        # things to keep in mind:
        # categorical data may have corupted entries in self.categoricalMap
        # make sure you update the header
        # make sure you update numcols
        # 1. delete the column
        for i in range(self.numrows):
            self.dataMatrix[i] = self.dataMatrix[i][:col] + self.dataMatrix[col + 1:]

        # 2. Update the header
        self.header = self.header[:col] + self.header[col + 1:]

        # 3. update numcols
        self.numcols -= 1

        # 4. un-corrput entries in categoricalMap. Note only entries corrupted are the ones that start at or after col.
        # cant change dict during iteration -- need to make a copy
        newMapping = {}

        for i, d in self.categoricalMap.items():
            if i < col:
                newMapping[i] = d
            if i > col:
                newMapping[i - 1] = d
        self.categoricalMap = newMapping

        # 5 update dtypes col
        self.dtypes = self.dtypes[:col] + self.dtypes[col + 1:]

    def getcol(self, col):
        '''
        Returns a column as a list from the datacontainer.
        :param col: index of column.
        :return: [header, column]
        '''
        assert col < self.numcols

        if col < 0:
            return self.getcol(self.numcols + col)

        header = self.header[col]
        col = [x[col] for x in self.dataMatrix]
        return header, col

    def writeCSV(self, fname):
        '''
        Write the DataContainer object to a csv file.
        :param fname: name of the target file.
        :return: None.
        '''
        f = open(fname, mode='w', newline='')

        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # first write the header, then every entry in the data matrix
        writer.writerow(self.header)

        # make a copy of a data matrix
        dm = deepcopy(self.dataMatrix)

        # need to reverse numerical categorical data to original names.
        for i in range(len(self.dtypes)):
            dtype = self.dtypes[i]
            if dtype == 'categorical':
                # here we need to change the values in our copy of the datamatrix
                for j in range(len(dm)):
                    dm[j][i] = self.findkey(self.dataMatrix[j][i], j, i)
        for r in dm:
            writer.writerow(r)

        print("Successfully wrote datacontainer to ", fname)

    #################################################################################
    #                               HELPER FUNCTIONS                                #
    #################################################################################
    def finddtype(self, col):
        '''
        For a given column, decides if the datatype is a number of categorical.
        :param col: index of column.
        :return: 'float' or 'categorical'
        '''

        c = self.getcol(col)[1]
        if c == '':
            return 'float'
            
        try:
            float(c[0])
        except:
            self.dtypes[col] = 'categorical'
            return 'categorical'

        self.dtypes[col] = 'float'
        return 'float'

    def mapcol(self, col):
        '''
        Maps the column to its appropriate datatype. Only called once when initializing.
        :param col: which column to map.
        :return: None, does mapping in-place.
        '''
        dtype = self.finddtype(col)

        # this is how we handle simple numbers
        if dtype == 'float':
            for i in range(len(self.dataMatrix)):
                try:
                    self.dataMatrix[i][col] = float(self.dataMatrix[i][col])
                except:
                    print("Wrong value inferred for ", self.dataMatrix[i][col], self.dtypes[col], 'on column', col, file=open('log', 'a'))
                    self.dataMatrix[i][col] = 0.0
        # categorical data might get a little more complicated
        else:
            # idea - find all categories
            # create dictionary mapping with an entry with the col number into categorical map
            # and update your data matrix
            # vals is maintains a one-hot encoding of the categories. The map is 'string' --> one-hot.
            vals = {}
            c = self.getcol(col)[1]
            for elem in c:
                if elem not in vals:
                    vals[elem] = len(vals)

            # update the data matrix
            for i in range(len(self.dataMatrix)):
                self.dataMatrix[i][col] = vals[self.dataMatrix[i][col]]

            # remember the encoding into categorical map
            self.categoricalMap[col] = vals

    def findkey(self, entry, row, col):
        '''
        Reverse maps number to categorical data for one entry/
        :param entry: the value of entry
        :param row: row at which entry occurs
        :param col: col at which entry occurs
        :return: reverse-mapped string of categorical data
        '''
        mapping = self.categoricalMap[col]
        return list(mapping.keys())[list(mapping.values()).index(entry)]


dc = DataContainer('test.csv')
dc1 = DataContainer('train.csv')
# print(dc)
# print(dc.getcol(3))
# print(dc.getcol(2))
# print(dc.getcol(0))
# sub = subsetDataContainer(dc, [23])
# sub.dropcol(0)
# print([x[0] for x in sub.dataMatrix])
