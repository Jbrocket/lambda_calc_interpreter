import string
import ast
import itertools
from collections import namedtuple

PUNCTUATION = ['\\', '.', '(', ')']
WHITESPACE = list(string.whitespace)

Token = namedtuple('Token', ['type', 'value'])


class Lexer(object):
    """An iterator that splits lambda calculus source code into Tokens.

    Attributes:
        source (str): Lambda calculus source code
        size (int): The number of characters in the source
        position (int): Current index in the source
    """

    def __init__(self, source):
        self.source = source
        self.size = len(source)
        self.position = 0

    def __iter__(self):
        return self

    def next(self):
        """Returns the next lexeme as a Token object."""
        self._clear_whitespace()
        if self.position > self.size:
            raise StopIteration()
        elif self.position == self.size:
            self.position += 1
            return Token('EOF', None)
        elif self.source[self.position] in PUNCTUATION:
            char = self.source[self.position]
            self.position += 1
            return Token(char, None)
        else:
            symbol = ''
            while (self.position < self.size and
                   not self.source[self.position] in PUNCTUATION + WHITESPACE):
                symbol += self.source[self.position]
                self.position += 1
            return Token('SYMBOL', symbol)

    def _clear_whitespace(self):
        """Advances position past any whitespace."""
        while (self.position < self.size and
               self.source[self.position] in string.whitespace):
            self.position += 1

class Expression(object):
    """Abstract class for any lambda calculus expression."""

    def children(self):
        """Returns a list of Expression objects."""
        pass

class FreeVariables(ast.NodeVisitor):
    """Visits each node of a lambda calculus abstract syntax tree and
    determines which variables (if any) are unbound. Ultimately provides a set
    of string variable names.
    """

    def visit_Variable(self, node):
        """FV(x) = {x}"""
        return {node.name}

    def visit_Application(self, node):
        """FV((e1 e2)) = FV(e1) U FV(e2)"""
        return (self.visit(node.left_expression) |
                self.visit(node.right_expression))

    def visit_Abstraction(self, node):
        """FV(\\x.e) = FV(e) - {x}"""
        return self.visit(node.body) - self.visit(node.parameter)


class BoundVariables(ast.NodeVisitor):
    """A variable is bound when a surrounding abstraction defines its scope.
    This visitor traverses a lambda calculus abstract syntax tree and provides
    a set of all bound variable names.
    """

    def visit_Variable(self, node):
        """BV(x) = {}"""
        return set()

    def visit_Application(self, node):
        """BV((e1 e2)) = BV(e1) U BV(e2)"""
        return (self.visit(node.left_expression) |
                self.visit(node.right_expression))

    def visit_Abstraction(self, node):
        """BV(\\x.e) = BV(e) U {x}"""
        return self.visit(node.body) | {node.parameter.name}


class AlphaConversion(ast.NodeVisitor):
    """Nondestructively substitutes all free occurances of a particular
    variable for an arbitrary expression.

    Attributes:
        to_return (Variable): Instance whose name attribute must match the
            variable that's being replaced
        replacement (Expression): Object inserted into the visited AST
    """

    def __init__(self, to_replace, replacement):
        self.to_replace = to_replace
        self.replacement = replacement

    def visit_Variable(self, node):
        """If the appropriate variable name is found, replace it."""
        if node.name == self.to_replace.name:
            return self.replacement
        else:
            return Variable(node.name)

    def visit_Application(self, node):
        """Returns a new Application after visiting both application
        expressions.
        """
        return Application(self.visit(node.left_expression),
                           self.visit(node.right_expression))

    def visit_Abstraction(self, node):
        """Returns a new Abstraction after visiting both the parameter and
        body. Renames the parameter if the replacement would be incorrectly
        bound by the abstraction.
        """
        if node.parameter.name in FreeVariables().visit(self.replacement):
            # name conflict with bound variable
            unavailable_names = (FreeVariables().visit(node) |
                                 {node.parameter.name})
            new_name = next(s for s in lexicographical()
                            if s not in unavailable_names)
            new_parameter = Variable(new_name)
            converter = AlphaConversion(node.parameter, new_parameter)
            new_body = converter.visit(node.body)
            return Abstraction(new_parameter, self.visit(new_body))
        else:
            return Abstraction(self.visit(node.parameter),
                               self.visit(node.body))


def lexicographical():
    """All alphabetic strings in lexicographical order."""
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    for size in itertools.count(1):
        for string in itertools.product(alphabet, repeat=size):
            yield ''.join(string)


class BetaReduction(ast.NodeVisitor):
    """Embodies the act of applying a function to a parameter. In operational
    semantics, (\\x.M N) → M[N/x]. This visitor provides a new abstract syntax
    tree with a single normal order reduction performed (if possible).

    Attributes:
        reduced (bool): Indicates if a reduction took place. If this variable
           remains false after a syntax tree is visited, the tree is in its
           normal form.
    """

    def __init__(self):
        self.reduced = False

    def visit_Variable(self, node):
        """Clones the given Variable node."""
        return Variable(node.name)

    def visit_Application(self, node):
        """Performs the application if the left-hand side represents an
        Abstraction and a reduction hasn't already taken place. Otherwise,
        the left-hand side and right-hand side are visited (in that order).
        """
        if (isinstance(node.left_expression, Abstraction) and
            not self.reduced):
            self.reduced = True
            converter = AlphaConversion(node.left_expression.parameter,
                                        node.right_expression)
            return converter.visit(node.left_expression.body)
        else:
            return Application(self.visit(node.left_expression),
                               self.visit(node.right_expression))

    def visit_Abstraction(self, node):
        """Returns a new Abstraction after visiting the parameter and body."""
        return Abstraction(self.visit(node.parameter),
                           self.visit(node.body))

class Variable(Expression):
    """Encapsulates a lambda calculus variable.

    Attributes:
        name (str): The variable's ID
    """

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def children(self):
        return []

class Parser(object):
    """An LL(1) parser that performs syntactic analysis on lambda calculus
    source code. An abstract syntax tree is provided if the given expression is
    valid.

    Attributes:
        lexer (Lexer): A tokenizer that's iteratively read
        token (Token): The current Token object
    """

    def __init__(self, lexer):
        self.lexer = lexer
        self.token = self.lexer.next()

    def _error(self, expected):
        """Raises a ParserError that compares an expected token type and the
        one given by the lexer.
        """
        raise ParserError(expected, self.token.type)

    def _advance(self):
        """Moves to the next token"""
        self.token = self.lexer.next()

    def _eat(self, prediction):
        """Advances through the source but only if type of the next token
        matches an expected form
        """
        if self.token.type == prediction:
            self._advance()
        else:
            self._error(prediction)

    def _expression(self):
        """Based on the current token, this method decides if the next
        expression is an application, abstraction, or variable"""
        if self.token.type == '(':
            return self._application()
        elif self.token.type == '\\':
            return self._abstraction()
        elif self.token.type == 'SYMBOL':
            return self._variable()
        else:
            self._error(u'(, \\, or SYMBOL')

    def _variable(self):
        """Returns an instance of Variable if the current token is a symbol"""
        if self.token.type == 'SYMBOL':
            name = self.token.value
            self._advance()
            return Variable(name)
        else:
            self._error('SYMBOL')

    def _application(self):
        """Returns an Application instance if the current toke is a left
        parenthesis
        """
        if self.token.type == '(':
            self._advance()
            left_expression = self._expression()
            right_expression = self._expression()
            self._eat(')')
            return Application(left_expression, right_expression)
        else:
            self._error('(')

    def _abstraction(self):
        """Returns an instance of Abstraction if the next series of tokens
        fits the form of a lambda calculus function"""
        if self.token.type == '\\':
            self._advance()
            variable = self._variable()
            self._eat('.')
            return Abstraction(variable, self._expression())
        else:
            self._error('\\')

    def parse(self):
        """Returns an abstract syntax tree if the source correctly fits the
        rules of lambda calculus
        """
        return self._expression()


class ParserError(Exception):
    """Indicates a discrepancy between what a parser expects and an actual
    value.

    Attributes:
        expected (str): The type that should have existed
        found (str): The actual type discovered
    """

    def __init__(self, expected, found):
        message = u'Expected: {}, Found: {}'.format(expected, found)
        super(ParserError, self).__init__(message)
        self.expected = expected
        self.found = found

class Application(Expression):
    """Encapsulates a lambda calculus function call.

    Attributes:
        left_expression (Expression): A function to be evaluated
        right_expression (Expression): The argument that's applied
    """

    def __init__(self, left_expression, right_expression):
        self.left_expression = left_expression
        self.right_expression = right_expression

    def __str__(self):
        return u'({} {})'.format(self.left_expression, self.right_expression)

    def children(self):
        return [self.left_expression, self.right_expression]


class Abstraction(Expression):
    """Encapsulates a function in lambda calculus.

    Attributes:
        parameter (Variable): The argument variable
        body (Expression): The scope of the function
    """

    def __init__(self, parameter, body):
        self.parameter = parameter
        self.body = body

    def __str__(self):
        return u'\\{}.{}'.format(self.parameter, self.body)

    def children(self):
        return [self.parameter, self.body]

def interpret(input_string, print_reductions=False):
    """Performs normal order reduction on the given string lambda calculus
    expression. Returns the expression's normal form if it exists.
    """
    lexer = Lexer(input_string)
    try:
        ast = Parser(lexer).parse()
    except ParserError as discrepancy:
        print('ParseError: ' + discrepancy.message)
        return None
    normal_form = False
    while not normal_form:
        reducer = BetaReduction()
        reduced_ast = reducer.visit(ast)
        normal_form = not reducer.reduced
        if print_reductions:
            print(ast)
        ast = reduced_ast
    return str(ast)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        interpret(sys.argv[1], print_reductions=True)
    else:
        while True:
            read = input('> ')
            if read == 'quit':
                break
            if read != '':
                interpret(read, print_reductions=True)