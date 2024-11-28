from bfcl.model_handler.oss_model.base_oss_handler import OSSHandler
from overrides import overrides


# Note: This is the handler for the Llama models in prompring mode.
# For function call mode, use LlamaFCHandler instead.
# Llama 3 series are benchmarked in prompting mode while the Llama 3.1 series are benchmarked in function call mode.
class LlamaHandler(OSSHandler):
    def __init__(self, model_name, temperature) -> None:
        super().__init__(model_name, temperature)

    @overrides
    def _format_prompt(self, messages, function):
        formatted_prompt = "<|begin_of_text|>"

        for message in messages:
            formatted_prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content'].strip()}<|eot_id|>"

        formatted_prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n"

        return formatted_prompt

    @overrides
    def decode_ast(self, result, language="Python"):
        try:
            result = result.replace("<|python_tag|>", "")
            # Llama sometimes separates the function calls with `;` and sometimes with `,`
            if result.startswith("[") and not result.endswith("]"):
                result = result + "]"
            if ";" in result:
                """
                "<|python_tag|>{\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"10\", \"k\": \"3\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"15\", \"k\": \"5\", \"p\": \"0\"}}; {\"name\": \"calc_binomial_probability\", \"parameters\": {\"n\": \"20\", \"k\": \"7\", \"p\": \"0\"}}"
                """
                function_calls = result.split(";")
                function_calls = [json.loads(func_call) for func_call in function_calls]
            elif "=" in result and "(" in result:
                res = super().decode_ast(result, language)
                return res
            else:
                """
                "[\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"20\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"12\", \"k\": \"5\"}},\n    {\"name\": \"calculate_permutations\", \"parameters\": {\"n\": \"10\", \"k\": \"3\"}}\n]"
                """
                if result.startswith("[") and not result.endswith("]"):
                    result = result + "]"
                function_calls = eval(result)
                if type(function_calls) == dict:
                    function_calls = [function_calls]

            decoded_output = []
            for func_call in function_calls:
                name = func_call["name"]
                params = func_call["parameters"]
                decoded_output.append({name: params})

            return decoded_output
        except Exception as e:
            raise e

    @overrides
    def decode_execute(self, result):
        print("decode execute")
        result = result.replace("<|python_tag|>", "")
        # Llama sometimes separates the function calls with `;` and sometimes with `,`
        if ";" in result:
            function_calls = result.split(";")
            function_calls = [json.loads(func_call) for func_call in function_calls]
        else:
            function_calls = eval(result)
            if type(function_calls) == dict:
                function_calls = [function_calls]

        execution_list = []
        for func_call in function_calls:
            name = func_call["name"]
            params = func_call["parameters"]
            execution_list.append(
                f"{name}({','.join([f'{k}={repr(v)}' for k,v in params.items()])})"
            )

        return execution_list