function P = EvaluateClassifierSigmoid(X, W, b)

    s = W*X + b;
    P = (1+exp(-s)).^-1;

end