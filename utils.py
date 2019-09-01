

def word_ids(texts, tokenizer, max_seq_length):
    input_ids_list = []  # 字符映射id
    input_mask_list = []  # 判断句子真实长度, 因为设置了sequence_length
    segment_ids_list = []  # 句子标记 bert可以判断是否为当前句的下一句  不需要此功能 可以直接设置为0
    for text in texts:
        text_list = text.split(' ')
        tokens = []
        for word in text_list:
            token = tokenizer.tokenize(word)  # 分字 中文默认 为字符
            tokens.extend(token)

        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]

        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")  # 添加句首 标记
        segment_ids.append(0)

        for token in tokens:
            ntokens.append(token)
            segment_ids.append(0)

        ntokens.append("[SEP]")  # 添加句尾标记
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)  # 字符转化为id
        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
    return input_ids_list, input_mask_list, segment_ids_list
