defmodule Bumblebee.Text.Generation.StackTest do
  use ExUnit.Case, async: true

  import Bumblebee.TestHelpers

  alias Bumblebee.Text.Generation.Stack

  describe "stack" do
    test "test stack" do
      stack = Stack.new(10)

      stack =
        stack
        |> Stack.push(100)
        |> Stack.push(99)

      value =
        stack
        |> Stack.peek()

      assert_equal(value, 99)

      len =
        stack
        |> Stack.stack_length()

      assert_equal(len, 2)

      {value, stack} =
        stack
        |> Stack.pop()

      assert_equal(value, 99)

      {value, stack} =
        stack
        |> Stack.pop()

      assert_equal(value, 100)

      len =
        stack
        |> Stack.stack_length()

      assert_equal(len, 0)
    end
  end
end
