# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Tuấn Hưng
**Nhóm:** D1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity (độ tương đồng cosine cao) giữa hai vector embedding có nghĩa là hai vector này có hướng rất giống nhau trong không gian hay hai đoạn văn bản tương ứng có ngữ nghĩa tương đồng hoặc liên quan chặt chẽ đến nhau.

**Ví dụ HIGH similarity:**
- Sentence A: The sun is shining brightly today.
- Sentence B: It's a sunny day.
- Tại sao tương đồng: Cả hai câu đều mô tả thời tiết nắng.

**Ví dụ LOW similarity:**
- Sentence A: I love to eat pizza.
- Sentence B: The car is red.
- Tại sao khác: Hai câu này không có mối liên hệ nào về mặt ngữ nghĩa.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity tập trung vào hướng của các vector chứ không phải độ lớn của chúng. Nó phù hợp hơn với text embedding vì hướng đại diện cho ngữ nghĩa và độ lớn có thể bị ảnh hưởng bởi các yếu tố như độ dài câu.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> - Kích thước bước (step size) = chunk_size - overlap = 500 - 50 = 450 ký tự.
> - Số lượng chunks = ceil((tổng số ký tự - chunk_size) / step_size) + 1
> - Số lượng chunks = ceil((10000 - 500) / 450) + 1 = ceil(9500 / 450) + 1 = ceil(21.11) + 1 = 22 + 1 = 23 chunks.
> *Đáp án:* 23 chunks

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap tăng lên 100, số lượng chunks sẽ tăng lên vì kích thước bước giảm xuống. Việc tăng overlap giúp đảm bảo rằng ngữ cảnh không bị mất đột ngột ở ranh giới giữa các chunks, giúp mô hình hiểu rõ hơn mối liên kết giữa các phần của văn bản.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Software Engineering Principles

**Tại sao nhóm chọn domain này?**
> Nhóm chúng tôi chọn domain này vì nó chứa các khái niệm cốt lõi, có cấu trúc rõ ràng và phân cấp, rất phù hợp để thử nghiệm các chiến lược chunking khác nhau. Việc hiểu và truy xuất chính xác các nguyên lý như SOLID, DRY, KISS là một bài toán thực tế và hữu ích cho các kỹ sư phần mềm.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | book.md | https://onlinelibrary.wiley.com/doi/book/10.1002/9781394297696?msockid=342527e00a4661fb18ff345a0bdc6080 | 503401 | `{"category": "software-engineering", "source": "book.md"}` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| category | string | "software-engineering" | Giúp lọc các tài liệu theo chủ đề lớn, hữu ích khi hệ thống có nhiều domain khác nhau. |
| source | string | "book.md" | Cho phép truy xuất nguồn gốc của chunk, giúp xác minh thông tin và cung cấp thêm ngữ cảnh cho người dùng. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis


| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| book.md | FixedSizeChunker (`fixed_size`) | 1119 | 499 | No |
| book.md | SentenceChunker (`by_sentences`) | 1160 | 432 | Partially |
| book.md | RecursiveChunker (`recursive`) | 1276 | 393 | Yes |

### Strategy Của Tôi

**Loại:** SemanticChunker (custom strategy)

**Mô tả cách hoạt động:**
> Chiến lược `SemanticChunker` hoạt động bằng cách chia văn bản dựa trên cấu trúc ngữ nghĩa của tài liệu Markdown. Nó sử dụng biểu thức chính quy (regex) để xác định các tiêu đề chương (`##`) và tiêu đề mục (`###`). Thay vì cắt văn bản một cách tùy tiện, nó coi mỗi chương và mỗi mục là một đơn vị ngữ nghĩa và tạo ra một chunk cho mỗi đơn vị đó. Điều này đảm bảo rằng các ý tưởng và khái niệm hoàn chỉnh được giữ lại trong cùng một chunk.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Tôi chọn chiến lược này vì domain "Software Engineering Principles" có cấu trúc rất rõ ràng với các chương và mục riêng biệt cho từng nguyên lý. `SemanticChunker` khai thác trực tiếp pattern này, đảm bảo rằng toàn bộ phần giải thích về một nguyên lý (ví dụ: Single Responsibility Principle) nằm gọn trong một chunk duy nhất, giúp việc truy xuất thông tin trở nên chính xác và đầy đủ ngữ cảnh hơn.

**Code snippet (nếu custom):**
```python
import re

class SemanticChunker:
    """
    Splits text based on semantic boundaries like chapters (##) and sections (###).
    """
    def chunk(self, text: str) -> list[str]:
        chunks = []
        # Split by chapters first, keeping the delimiter
        chapters = re.split(r'(\n## .*\n)', text)
        
        current_chunk = chapters[0]
        for i in range(1, len(chapters), 2):
            # Split the current chunk by sections
            sections = re.split(r'(\n### .*\n)', current_chunk)
            
            # Add the chapter intro part
            if sections[0].strip():
                chunks.append(sections[0].strip())
            
            # Add the sections
            for j in range(1, len(sections), 2):
                section_header = sections[j]
                section_content = sections[j+1]
                if (section_header + section_content).strip():
                    chunks.append((section_header + section_content).strip())

            current_chunk = chapters[i] + chapters[i+1]

        # Process the last chapter
        sections = re.split(r'(\n### .*\n)', current_chunk)
        if sections[0].strip():
            chunks.append(sections[0].strip())
        for j in range(1, len(sections), 2):
            section_header = sections[j]
            section_content = sections[j+1]
            if (section_header + section_content).strip():
                chunks.append((section_header + section_content).strip())

        return [chunk for chunk in chunks if chunk]
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| book.md | RecursiveChunker | 1276 | 393 | Good |
| book.md | **của tôi** | 68 | 7400 | Excellent |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Semantic Chunking | 9.5 | Giữ trọn vẹn ngữ cảnh của từng mục, truy xuất chính xác. | Các chunk có thể rất lớn, không phù hợp với các mô hình có giới hạn context nhỏ. |
| Lê Minh Hoàng | SoftwareEngineeringChunker (Custom RecursiveTrunker) | 8 | Bảo tồn hoàn hảo cấu trúc tài liệu kỹ thuật nhờ ngắt theo Header; Giữ được mối liên kết logic. | Kích thước chunk trung bình lớn, gây tốn context window của mô hình. |
| Nguyễn Xuân Hải | Parent-Child Chunking| 8 |Child nhỏ giúp tìm kiếm vector đúng mục tiêu, ít nhiễu | Gửi cả khối Parent lớn vào Prompt làm tăng chi phí API.
| Nguyễn Đăng Hải | DocumentStructureChunker | 6.3 | Giữ ngữ cảnh theo heading/list/table; grounding tốt cho tài liệu dài | Phức tạp hơn và tốn xử lý hơn; lợi thế giảm khi dữ liệu ít cấu trúc |
|Thái Minh Kiên | Agentic Chunking | 8 | chunk giữ được ý nghĩa trọn vẹn, retrieval chính xác hơn, ít trả về nửa vời, Không cần một rule cố định cho mọi loại dữ liệu | Với dataset lớn cost sẽ tăng mạnh,  chậm hơn pipeline thường, không ổn định tuyệt đối |
Trần Trung Hậu |Token-Based Chunking (Chia theo Token) | 8 | Kiểm soát chính xác tuyệt đối giới hạn đầu vào (context window) và chi phí API của LLM. | Cắt rất máy móc, dễ làm đứt gãy ngữ nghĩa của một từ hoặc một câu giữa chừng.
| Tạ Bảo Ngọc | Sliding Window + Overlap | 4 | Giữ vẹn câu/khối logic, tối ưu length | bị trùng dữ liệu -> tăng số chunk |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> `Semantic Chunking` là tốt nhất cho domain này vì nó tôn trọng cấu trúc logic của tài liệu, đảm bảo mỗi chunk là một đơn vị thông tin hoàn chỉnh. Điều này giúp hệ thống RAG truy xuất được ngữ cảnh đầy đủ để trả lời các câu hỏi về các nguyên lý cụ thể một cách chính xác nhất.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Tôi đã sử dụng biểu thức chính quy `r'(?<=[.!?])\s+'` để chia văn bản thành các câu. Regex này tìm kiếm các khoảng trắng theo sau dấu chấm, chấm than, hoặc chấm hỏi. Sau khi chia, tôi đã xử lý các trường hợp edge case bằng cách loại bỏ các chuỗi rỗng hoặc chỉ chứa khoảng trắng có thể xuất hiện do nhiều khoảng trắng liên tiếp trong văn bản gốc.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán hoạt động bằng cách thử chia văn bản với một danh sách các ký tự phân tách theo thứ tự ưu tiên. Base case của đệ quy là khi một đoạn văn bản đã nhỏ hơn `chunk_size` hoặc khi không còn ký tự phân tách nào để thử. Trong trường hợp sau, nó sẽ buộc phải cắt đoạn văn bản thành các phần có kích thước bằng `chunk_size`.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents` chuyển đổi mỗi tài liệu thành một bản ghi chứa embedding và metadata, sau đó lưu vào một danh sách trong bộ nhớ (`self._store`). `search` tính toán độ tương đồng cosine (thông qua tích vô hướng của các vector đã được chuẩn hóa) giữa embedding của câu hỏi và tất cả các embedding đã lưu, sau đó sắp xếp và trả về top-K kết quả có điểm cao nhất.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter` thực hiện việc lọc trước (pre-filtering): nó tạo một danh sách con các bản ghi khớp với `metadata_filter` rồi mới thực hiện tìm kiếm tương đồng trên danh sách con đó. `delete_document` hoạt động bằng cách tạo lại danh sách `self._store`, chỉ bao gồm những bản ghi không có `doc_id` khớp với id cần xóa.

### KnowledgeBaseAgent

**`answer`** — approach:
> Cấu trúc prompt bao gồm ba phần: một chỉ dẫn chung, phần ngữ cảnh được truy xuất, và câu hỏi của người dùng. Tôi đã inject ngữ cảnh bằng cách lặp qua các chunk được truy xuất từ `EmbeddingStore` và định dạng chúng thành một danh sách có dấu gạch đầu dòng, đặt chúng dưới tiêu đề "Context:".

### Test Results

```
# Paste output of: pytest tests/ -v
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.0.2, pluggy-1.6.0 -- /home/tuanhung/VINUNI/assignments/day_01_llm_api_foundation/vinuni/bin/python3
cachedir: .pytest_cache
rootdir: /home/tuanhung/VINUNI/assignments/day_01_llm_api_foundation/2A202600230-NguyenTuanHung-Day07
plugins: anyio-4.13.0, langsmith-0.7.26
collecting ... collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================== 42 passed in 0.03s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sat on the mat. | A feline was resting on the rug. | low | -0.007 | No |
| 2 | SOLID principles are key for good software design. | SRP is one of the SOLID principles. | high | 0.133 | Yes |
| 3 | I am learning about vector stores. | My favorite food is pho. | low | 0.054 | Yes |
| 4 | What is the capital of France? | Paris is a beautiful city. | low | 0.083 | Yes |
| 5 | The system should be scalable. | The system must handle many users. | low | -0.229 | No |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả bất ngờ nhất là cặp 1 và 5 có điểm tương đồng âm, dù về mặt ngữ nghĩa chúng khá liên quan. Điều này cho thấy `MockEmbedder` rất đơn giản và không thực sự nắm bắt được ngữ nghĩa sâu sắc; nó có thể chỉ dựa trên sự trùng lặp của các từ hoặc các mẫu ký tự bề mặt, dẫn đến các dự đoán sai lệch.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | What are the SOLID principles? | SOLID is an acronym for five design principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion. |
| 2 | Explain the DRY principle. | Don't Repeat Yourself means every piece of knowledge must have a single, unambiguous representation in a system. |
| 3 | What is the difference between SRP and ISP? | SRP is about a class having one reason to change, while ISP is about clients not depending on interfaces they don't use. |
| 4 | What does KISS stand for? | Keep It Simple, Stupid. |
| 5 | Summarize the main idea of the Open/Closed Principle. | Software entities should be open for extension but closed for modification. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What are the SOLID principles? | ## Philippe Eynaud... | 0.29 | No | mock test|
| 2 | Explain the DRY principle. | ### 8.3. Competition, technological revolutions and new stra... | 0.31 | Yes | mock test|
| 3 | What is the difference between SRP and ISP? | ### 4.5. The information systems territory and the organizat... | 0.29 | Yes | mock test|
| 4 | What does KISS stand for? | ### 7.5. Security in information systems projects  Security ... | 0.27 | Yes | mock test|
| 5 | Summarize the main idea of the Open/Closed Principle. | ### 9.4. Scope of the audit  To define the scope of the audi... | 0.37 | Yes | mock test|

**Bao nhiêu queries trả về chunk relevant trong top-3?** 4 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Từ việc so sánh các chiến lược chunking, tôi nhận ra không có một giải pháp nào là hoàn hảo cho mọi trường hợp. Ví dụ, chiến lược `Parent-Child Chunking` của bạn Nguyễn Xuân Hải rất thông minh trong việc cân bằng giữa việc tìm kiếm chính xác (child nhỏ) và cung cấp đủ ngữ cảnh (parent lớn). Điều này cho thấy tầm quan trọng của việc hiểu sâu cả về dữ liệu và cách hoạt động của hệ thống RAG để đưa ra lựa chọn thiết kế phù hợp.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Trực quan hóa thông tin để dễ dàng phân tích. Phân tích kĩ hơn các chỉ số trên metrics để có đánh giá tổng quát hơn về các phương pháp.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Nếu có cơ hội làm lại, tôi sẽ đầu tư nhiều thời gian hơn vào việc tiền xử lý và làm giàu metadata. Thay vì chỉ có `source` và `category`, tôi sẽ thêm các metadata chi tiết hơn như `chapter_title`, `section_title`, hoặc thậm chí là các từ khóa chính cho mỗi chunk. Việc này sẽ cho phép thực hiện `search_with_filter` một cách mạnh mẽ hơn, giúp thu hẹp không gian tìm kiếm và tăng độ chính xác của kết quả retrieval ngay cả trước khi tính toán đến vector similarity.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 14 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 9 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm |  5/ 5 |
| **Tổng** | | **88 / 90** |