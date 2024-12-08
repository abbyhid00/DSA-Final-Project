import copy
import csv
from tabulate import tabulate

"""
Name: Zobe Murray
Course: CPSC 322
Date: 10/29/2024
Description: This program creates MyPyTable objects that store data loaded in from csv files. There different 
functions that the class has built in (get_shape, get_column, etc.) that inform users about the data and enable 
two MyPyTable objects to be merged.

"""

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        return len(self.data), len(self.column_names)

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        new_values = []
        if col_identifier not in self.column_names:
            raise ValueError(f"{col_identifier} is not valid")
        col_index = self.column_names.index(col_identifier)
        for row in self.data:
            new_values.append(row[col_index])
        return new_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row_index, row in enumerate(self.data):
            for col_index, value in enumerate(row):
                #try to convert every value in row to float if you cannot throw TypeError/ValueError
                try:
                    self.data[row_index][col_index] = float(value)
                #When exception thrown pass the value
                except (TypeError, ValueError):
                    pass
        return self

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Args:
            row_indexes_to_drop(list of int): list of row indexes to remove from the table data.
        """
        new_table = []
        for row in range(len(self.data)):
            #If the row is not in the indexes to drop add the row to the new_table list
            if row not in row_indexes_to_drop:
                new_table.append(self.data[row])
        #Replace the pytable with the new table data
        self.data = new_table
        return self

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        #opening and reading file
        infile = open(filename, 'r', encoding= 'utf-8')
        reader = csv.reader(infile)
        #Save first row/header in the object's column name list
        self.column_names = next(reader)
        #iterating through the table and adding each row to the table
        for row in reader:
            self.data.append(row)
        #closing the file
        infile.close()
        #Converting the number values to floats
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        #Opening the file that will be written on
        outfile = open(filename, 'w', newline='')
        writer = csv.writer(outfile)
        #Write the header
        writer.writerow(self.column_names)
        #Write the data
        writer.writerows(self.data)
        #Closing file
        outfile.close()
        return

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
                The first instance of a row is not considered a duplicate.
        """
        dup = []
        seen = []
        #Create table with just the columns we are interested in
        temp = self.extract_key(key_column_names)
        #Go through each row adding the first instance index to duplicate list
        #And adding the repeat rows to the seen list
        for i, row in enumerate(temp.data):
            if row in seen:
                dup.append(i)
            else:
                seen.append(row)

        return dup

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        new_table = []
        #Go through each row adding the ones without NA to the new table list
        for row in self.data:
            if "NA" not in row:
                new_table.append(row)
        #Replace the object data with the new data
        self.data = new_table
        return self

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
            by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        #Get the index of the column of interest
        col_index = self.column_names.index(col_name)
        non_missing = []
        #Going through each row at the column index and if it is not missing it
        #Is added to the non_missing list
        for row in self.data:
            if row[col_index] != "NA":
                non_missing.append(float(row[col_index]))
        #calculating the average from the list of non_missing values
        if non_missing:
            col_avg = sum(non_missing) / len(non_missing)
        #Going through column replacing NA with column average
            for row in self.data:
                if row[col_index] == "NA":
                    row[col_index] = col_avg
        return self

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Args:
            col_names(list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values should in the columns to compute summary stats
                for should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        #Creating stats table to hold the stat values
        stats = MyPyTable(column_names= ["attribute", "min", "max", "mid", "avg", "median"])
        #If there are not columns return
        if not col_names:
            return stats
        for i in range(len(col_names)):
            #for every column of interest get the name of the
            #column for attribute column
            name = col_names[i]
            #get column of interest
            col_list = self.get_column(name, False)
            #making sure that the values are floats
            numeric_values = []
            for j in range(len(col_list)):
                value = col_list[j]
                if value != "NA":
                    numeric_values.append(float(value))
            if numeric_values:
                #calculate stats
                minimum = min(numeric_values)
                maximum = max(numeric_values)
                mid = (minimum + maximum) / 2
                mean = sum(numeric_values) / len(numeric_values)
                #calling get_med function to calculate the median of the column
                median = self.get_med(numeric_values)
            #If the value is not a string continue
            else:
                continue
            #add the row to the data
            stats.data.append([name, minimum, maximum, mid, mean, median])
        return stats

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
            with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        #get columns in other table not in self table
        table2_col = []
        for col in other_table.column_names:
            if col not in key_column_names:
                table2_col.append(col)
        #creat new table with columns from self and other tables
        table3 = MyPyTable(self.column_names + table2_col)
        for row1 in self.data:
            #collecting the key values in the self table
            key_values1 = []
            for key_col in key_column_names:
                key_index1 = self.column_names.index(key_col)
                key_values1.append(row1[key_index1])
            #creating tuple of the values in key columns
            key_values1 = tuple(key_values1)
            for row2 in other_table.data:
                #collecting the key values in the other table
                key_values2 = []
                for key_col in key_column_names:
                    key_index2 = other_table.column_names.index(key_col)
                    key_values2.append(row2[key_index2])
                #creating tuple of the values in key columns
                key_values2 = tuple(key_values2)
                #for every row of the key values that match from the two tables
                #Copy the row portion from the self table
                #then add each column from the other table
                if key_values1 == key_values2:
                    joined_row = row1.copy()
                    for col in table2_col:
                        col_index = other_table.column_names.index(col)
                        joined_row.append(row2[col_index])
                    #add each row to the new table
                    table3.data.append(joined_row)
        return table3

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
            other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        #create the inner join portion of the table
        inner = self.perform_inner_join(other_table, key_column_names)
        #Extract key columns from the inner, self, and other tables
        inner_key = inner.extract_key(key_column_names)
        table1_key = self.extract_key(key_column_names)
        table2_key = other_table.extract_key(key_column_names)
        rows_indexes_add1 = []
        rows_indexes_add2 = []
        extra_col = 0
        #Find length of extra columns in the other table that are not in self
        for i in range(len(inner.column_names)):
            if inner.column_names[i] not in self.column_names:
                extra_col += 1
        #For each key column of self if it is not in inner add it to the
        #List of row indexes to add
        for j in range(len(table1_key.data)):
            is_missing = True
            for i in range(len(inner_key.data)):
                if inner_key.data[i] == table1_key.data[j]:
                    is_missing = False
                    break
            if is_missing:
                rows_indexes_add1.append(j)
        for row in rows_indexes_add1:
            #Add the self table only rows padding with NA in the
            #Other table exclusive columns
            new_row = self.data[row] + ["NA"] * extra_col
            inner.data.append(new_row)
        #For each key column of other if it is not in inner add it to the
        #List of row indexes to add
        for j in range(len(table2_key.data)):
            is_missing = True
            for i in range(len(inner_key.data)):
                if inner_key.data[i] == table2_key.data[j]:
                    is_missing = False
                    break
            if is_missing:
                rows_indexes_add2.append(j)
        other_to_inner_col = []
        #Finding where the other table values should be added in the
        #inner table
        for other_col in other_table.column_names:
            if other_col in inner.column_names:
                other_to_inner_col.append(inner.column_names.index(other_col))
            else:
                other_to_inner_col.append(None)
        for row in rows_indexes_add2:
            #Adding other table values padding the self exclusive
            #columns with NA
            new_row = ["NA"] * len(inner.column_names)
            for i, value in enumerate(other_table.data[row]):
                col_index = other_to_inner_col[i]
                if col_index is not None:
                    new_row[col_index] = value
            #add new row to the inner table
            inner.data.append(new_row)
        return inner

    def extract_key(self, col_names):
        """
        Returns new MyPyTable with the key columns only.
        Args:
            col_names(list): list of the key column names
        Returns:
            MyPyTable: table with data of the key columns
        """
        col_index = []
        #Create table with key column names as the column_names
        temp = MyPyTable(col_names)
        #Getting list of the key indexes
        for name in col_names:
            col_index.append(self.column_names.index(name))
        #Adding each column of key to the new table
        for row in self.data:
            new_row = []
            for index in col_index:
                new_row.append(row[index])
            temp.data.append(new_row)
        return temp

    def get_med(self, col_list):
        """
        Returns the median value of a list.
        Args:
            col_list(list): list of a column's values
        Returns:
            float: median value of the list
        """
        #Sorting the list
        temp = sorted(col_list)
        #Find median index of the list
        mid = len(temp) // 2
        #If the list has an even number of elements, returns the average of middle elements
        if len(temp) % 2 == 0:
            return (float(temp[mid - 1]) + float(temp[mid])) / 2
        #If the list has an even number of elements it just returns middle index value
        else: 
            return float(temp[mid])

    def get_rows(self, row_indexes):
        """Gets the tables rows at the given indexes.

        Args:
            row_indexes(list): indexes of the rows we want to return

        Returns: 
            rows(MyPyTable): table with the rows at the given indexes
        """
        #initialize table of rows we will return
        rows = MyPyTable(self.column_names)
        #go through find the rows we are interested in and add them to
        #the new tables data
        for index in row_indexes:
            for i, row in enumerate(self.data):
                if i == index:
                    rows.data.append(row)
        return rows

    def get_col_for_pred(self, col_of_interest):
        """Gets the tables features for the columns given.

        Args:
            col_of_interest(list): names of the columns we want the features of

        Returns: 
            data_of_interest(list of lists): list of features from the columns of interest for each instance
        """
        data_of_interest = []
        #go through each instance adding the adding the value if column of
        #interest to the instance's data and add it to the data of
        for i in range(len(self.data)):
            instance_data = []
            for col in col_of_interest:
                instance_data.append(self.get_column(col)[i])
            data_of_interest.append(instance_data)
        return data_of_interest
