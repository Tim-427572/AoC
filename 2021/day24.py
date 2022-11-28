#from rich import print
#from rich.traceback import install

#install()


def compile(filename):
    blocks = []
    instructions = []
    for line in open(filename):
        op = line.strip().split()
        if op[0] == "inp":
            instructions = []
            blocks.append(instructions)
        else:
            if len(op) == 3 and op[2] not in ("w", "x", "y", "z"):
                op[2] = int(op[2])
            instructions.append(op)
    return blocks


def sign(a):
    return 1 if a > 0 else -1 if a < 0 else 0


def div(x, y):
    return abs(x) // abs(y) * sign(x) * sign(y)


assert div(5, -3) == -1


def mod(x, y):
    return x - div(x, y) * y


assert mod(5, -3) == 2


def simplify(instructions, w, x, y, z, trace=False):
    regs = {"w": w, "x": x, "y": y, "z": z}
    for op, left, right in instructions:
        a = regs[left]
        b = right if isinstance(right, int) else regs[right]
        if op == "add":
            if isinstance(a, int) and isinstance(b, int):
                regs[left] = a + b
            elif b == 0:
                regs[left] = a
            elif a == 0:
                regs[left] = b
            else:
                regs[left] = ("add", a, b)
        elif op == "mul":
            if isinstance(a, int) and isinstance(b, int):
                regs[left] = a * b
            elif a == 0 or b == 0:
                regs[left] = 0
            elif a == 1:
                regs[left] = b
            elif b == 1:
                regs[left] = a
            else:
                regs[left] = ("mul", a, b)
        elif op == "div":
            if isinstance(a, int) and isinstance(b, int):
                regs[left] = div(a, b)
            elif a == 0:
                regs[left] = 0
            elif b == 1:
                regs[left] = a
            else:
                regs[left] = ("div", a, b)
        elif op == "mod":
            if isinstance(a, int) and isinstance(b, int):
                regs[left] = mod(a, b)
            elif a == 0:
                regs[left] = 0
            elif b == 1:
                regs[left] = 0
            else:
                regs[left] = ("mod", a, b)
        elif op == "eql":
            if isinstance(a, int) and isinstance(b, int):
                regs[left] = int(a == b)
            elif (
                isinstance(b, int)
                and (b < 1 or b > 9)
                and isinstance(left, str)
                and left.startswith("w")
            ):
                regs[left] = 0
            elif (
                isinstance(a, int)
                and (a < 1 or a > 9)
                and isinstance(right, str)
                and right.startswith("w")
            ):
                regs[left] = 0
            elif b == 0 and isinstance(a, tuple) and a[0] == "eql":
                regs[left] = ("neq", a[1], a[2])
            else:
                regs[left] = ("eql", a, b)
        else:
            raise RuntimeError(f"Invalid {op} {left} {right} {a} {b}")
        if trace:
            print((op, left, right, regs))
    return regs["x"], regs["y"], regs["z"]


def to_string(z):
    if isinstance(z, (int, str)):
        return str(z)
    op, a, b = z
    if op == "neq":
        return f"({to_string(a)}!={to_string(b)})"
    if op == "eql":
        return f"({to_string(a)}=={to_string(b)})"
    if op == "add":
        if isinstance(b, int) and b < 0:
            return f"({to_string(a)}-{-b})"
        return f"({to_string(a)}+{to_string(b)})"
    if op == "mul":
        return f"{to_string(a)}*{to_string(b)}"
    if op == "div":
        return f"({to_string(a)}//{to_string(b)})"
    if op == "mod":
        return f"({to_string(a)}%{to_string(b)})"


def z_target(block):
    for op, a, b in block:
        if op == "add" and a == "x" and isinstance(b, int):
            return -b if b < 0 else None
    return None


def part1(filename):
    blocks = compile(filename)
    states = {}
    result = {}
    #for w in range(1, 10):
    for w in range(9,0,-1):
        x, y, z = simplify(blocks[0], w, 0, 0, 0)
        result[z] = (w,)
    print(f"Step 0 z={min(result)}..{max(result)} {len(result)}")

    for index, b in enumerate(blocks[1:], start=1):
        prev = result
        result = {}
        dec = 0
        z_targ = z_target(b)
        #for w in range(1, 10):
        for w in range(9,0,-1):
            for z in prev:
                if z_targ is not None and z % 26 - z_targ != w:
                    continue
                x, y, z1 = simplify(b, w, 0, 0, z)
                if z1 not in result:
                    result[z1] = prev[z] + (w,)
                    if z1 < z:
                        dec += 1
        print(to_string(simplify(b, "w", "x", "y", "z")[2]))
        print(f"Step {index} z={min(result)}..{max(result)} {len(result)} {dec}")

    if not 0 in result:
        breakpoint()
    return "".join(str(n) for n in result[0])


print("Part 1", part1("day24.txt"))
