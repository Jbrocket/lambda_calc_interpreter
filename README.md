# Lambda Calculus Interpreter
This is a simple project that makes beta reductions on lambda calculus problems to simplify them

## Here's how to run the interpreter:

There are two ways to run the interpreter

for a one time calculation:
```sh
python src/interpreter.py <lambda_expression>
```

to enter a terminal where you can type expressions to be interpreted:
```sh
python src/interpreter.py
```

For our sake, lambdas are expression where defined inputs aren't sequential and are separated 
like this: "\x.\y.xy" rather than "\xy.xy". And, isntead of using a lambda symbol, a '\' is used instead