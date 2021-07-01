"These macros are taken and modified from https://github.com/autodiff/autodiff"

def declare_variables(variables, macro):
    """
    This is the hook for the functions

    - variables: the dictionary that contains the variables
    - macro: a decorator function, to declare a macro.
    """

    @macro
    def inputcode(filename, language, linenumbers=False, startline=0, endline=None):
        filename = '../' + filename  # file path must be given relative to root directory
        f = open(filename, 'r')
        if startline != 0 or endline != None:
            lines = f.readlines()
            lines = lines[startline:endline]
            text = "".join(lines)
        else:
            text = f.read()
        if linenumbers == False:
            textblock = f'```{{ .{language} .annotate}}\n{text}\n```'
        else:
            textblock = f'```{{ .{language} linenums="1" .annotate}} \n{text}\n```'
        return textblock

    @macro
    def inputcpp(filename,linenumbers=False, startline=0, endline=None):
        return inputcode(filename, 'cpp', linenumbers, startline, endline)
