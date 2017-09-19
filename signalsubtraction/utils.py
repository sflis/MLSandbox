import matplotlib.pyplot as plt
import string

header =  "+-------------------+----------------------------------------------------------------+\n"
header += "| Description       | Plot                                                           |\n"
header += "+===================+================================================================+\n"

def fill_char(char,n):
    return "".join([char for n in range(n)])

def as_rest_table(data, full=False):
    """
    >>> from report_table import as_rest_table
    >>> data = [('what', 'how', 'who'),
    ...         ('lorem', 'that is a long value', 3.1415),
    ...         ('ipsum', 89798, 0.2)]
    >>> print as_rest_table(data, full=True)
    +-------+----------------------+--------+
    | what  | how                  | who    |
    +=======+======================+========+
    | lorem | that is a long value | 3.1415 |
    +-------+----------------------+--------+
    | ipsum |                89798 |    0.2 |
    +-------+----------------------+--------+

    >>> print as_rest_table(data)
    =====  ====================  ======
    what   how                   who   
    =====  ====================  ======
    lorem  that is a long value  3.1415
    ipsum                 89798     0.2
    =====  ====================  ======

    """
    data = data if data else [['No Data']]
    table = []
    # max size of each column
    sizes = map(max, zip(*[[len(str(elt)) for elt in member]
                           for member in data]))
    num_elts = len(sizes)

    if full:
        start_of_line = '| '
        vertical_separator = ' | '
        end_of_line = ' |'
        line_marker = '-'
    else:
        start_of_line = ''
        vertical_separator = '  '
        end_of_line = ''
        line_marker = '='

    meta_template = vertical_separator.join(['{{{{{0}:{{{0}}}}}}}'.format(i)
                                             for i in range(num_elts)])
    template = '{0}{1}{2}'.format(start_of_line,
                                  meta_template.format(*sizes),
                                  end_of_line)
    # determine top/bottom borders
    if full:
        to_separator = string.maketrans('| ', '+-')
    else:
        to_separator = string.maketrans('|', '+')
    start_of_line = start_of_line.translate(to_separator)
    vertical_separator = vertical_separator.translate(to_separator)
    end_of_line = end_of_line.translate(to_separator)
    separator = '{0}{1}{2}'.format(start_of_line,
                                   vertical_separator.join([x*line_marker for x in sizes]),
                                   end_of_line)
    # determine header separator
    th_separator_tr = string.maketrans('-', '=')
    start_of_line = start_of_line.translate(th_separator_tr)
    line_marker = line_marker.translate(th_separator_tr)
    vertical_separator = vertical_separator.translate(th_separator_tr)
    end_of_line = end_of_line.translate(th_separator_tr)
    th_separator = '{0}{1}{2}'.format(start_of_line,
                                      vertical_separator.join([x*line_marker for x in sizes]),
                                      end_of_line)
    # prepare result
    table.append(separator)
    # set table header
    titles = data[0]
    table.append(template.format(*titles))
    table.append(th_separator)

    for d in data[1:-1]:
        table.append(template.format(*d))
        if full:
            table.append(separator)
    table.append(template.format(*data[-1]))
    table.append(separator)
    return '\n'.join(table)

def make_title(title,tag,template):
    s = ''
    s += title+"\n"
    s +=  "".join(['=' for ss in range(len(title))])+"\n"
    t = template.find(tag)
    return template[:t-1] + s +template[t+len(tag)+1:]

def insert(string,tag,template):
    t = template.find(tag)
    return template[:t-1] + string +template[t+len(tag)+1:]

class PlotTablePublisher(object):
    def __init__(self,title,col1 = 30,col2=61):
        self.title = title
        self.figures = list()
        self.col1 = col1
        self.col2 = col2
    
    def add_figures(self,fig_path,descr):
        self.figures.append([fig_path,descr])

    def publish(self):
        #col1 = 30
        #col2 = 61
        #t = template.find(tag)
        col1max = 0
        col2max = 0
        prefix_l = len(".. image:: ")
        for f in self.figures:
            if(len(f[0])+prefix_l>self.col2):
                self.col2 = prefix_l+len(f[0])
            if(len(f[1])>self.col1):
                self.col1 = len(f[1])
        s = ""
        #s += self.title+"\n"
        #s +=  "".join(['=' for ss in range(len(self.title))])+"\n"
        s += "+"+fill_char('-',self.col1)+'+'+fill_char('-',self.col2)+'+\n'
        s += ("| %-"+str(self.col1-1)+"s| %-"+str(self.col2-1)+"s|\n")%("Description","Plot")
        s += "+"+fill_char('=',self.col1)+'+'+fill_char('=',self.col2)+'+\n'
        #s += "+-------------------+----------------------------------------------------------------+\n"
        #s += "| Description       | Plot                                                           |\n"
        #s += "+===================+================================================================+\n"
        # print(self.figures)
        for f in self.figures:
            #print("one",f)
            #print("two",f[0])
            #print("three",f[1])
            s +=("|%-"+str(self.col1)+"s|%-"+str(self.col2)+"s|\n")%(f[1][:self.col1],".. image:: "+f[0])
            s +=("|%-"+str(self.col1)+"s|   %-"+str(self.col2-3)+"s|\n")%("",":width: 400px")
            s +=("|%-"+str(self.col1)+"s|   %-"+str(self.col2-3)+"s|\n")%("",":height: 300px")
            s += "+"+fill_char('-',self.col1)+'+'+fill_char('-',self.col2)+'+\n'
        #s += "+"+fill_char('=',self.col1)+'+'+fill_char('=',self.col2)+'+\n'


        return s#template[:t] + s +template[t+len(tag):]
