use rig::{
    client::{CompletionClient, EmbeddingsClient, ProviderClient},
    completion::Prompt,
    embeddings::EmbeddingsBuilder,
    providers::openai::Client,
    vector_store::in_memory_store::InMemoryVectorStore,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 准备文档
    let documents = vec![
        "Rig 是一个用于构建 LLM 驱动的应用的 Rust 库。",
        "RAG 结合了检索和生成以提高准确性。",
        "向量存储支持文档的语义搜索。",
        "DashScope 是阿里云提供的大模型 API 服务。",
    ];

    // 2. 使用独立的客户端创建嵌入模型和向量索引
    let embedding_client = Client::from_env();
    let embedding_model = embedding_client.embedding_model("text-embedding-v4");

    let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
        .documents(documents)?
        .build()
        .await?;

    let mut vector_store = InMemoryVectorStore::default();
    vector_store.add_documents(embeddings);
    let vector_index = vector_store.index(embedding_model);

    // 3. 转换为 Completions API 以兼容 DashScope，创建对话 agent
    let completions_client = Client::from_env().completions_api();
    let agent = completions_client
        .agent("qwen-plus")
        .preamble("你是一个有用的 AI 助手，根据提供的上下文回答问题。")
        .dynamic_context(2, vector_index) // 检索 2 个最相关的文档
        .build();

    // 4. 提问
    let response = agent.prompt("Rig 是什么？它如何帮助 LLM 应用？").await?;

    println!("{}", response);
    Ok(())
}
