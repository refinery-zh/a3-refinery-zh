from collections import deque

class A:
    def __init__(self) -> None:
        self.a = "1"

if __name__ == "__main__":
    queue = deque()
    a_list = [A() for _ in range(10)]
    for each in a_list:
        queue.append(each)
    a_list[1].a = "2"
    print(a_list)
    while len(queue):
        ele = queue.popleft()
        print(ele.a)