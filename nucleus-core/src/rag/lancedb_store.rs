//! LanceDB vector database storage implementation.
//!
//! This module provides integration with LanceDB for embedded, in-process vector storage.

use crate::config::RagConfig;

use super::store::VectorStore;
use super::types::{Document, SearchResult};
use anyhow::{Context, Result};
use lancedb::arrow::arrow_schema::{DataType, Field, Schema};
use arrow_array::{
    array::{ArrayRef, FixedSizeListArray, Float32Array, StringArray},
    Array, RecordBatch, RecordBatchIterator,
};
use futures::stream::TryStreamExt;
use async_trait::async_trait;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection, Table};
use std::sync::Arc;

/// LanceDB-based vector store for embedded deployment.
///
/// Provides zero-setup, in-process vector storage using LanceDB.
pub struct LanceDbStore {
    config: RagConfig, 
    conn: Connection,
    table: Table,
    vector_size: u64,
}

#[async_trait]
impl VectorStore for LanceDbStore {
    async fn add(&self, document: Document) -> Result<()> {
        let schema = Self::create_schema(self.vector_size);

        let id_array = StringArray::from(vec![document.id.as_str()]);
        let content_array = StringArray::from(vec![document.content.as_str()]);
        
        let vector_values = Float32Array::from(document.embedding);
        let vector_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            self.vector_size as i32,
            Arc::new(vector_values),
            None,
        );
        
        let source_value = document.metadata.get("source").map(|s| s.as_str());
        let source_array = StringArray::from(vec![source_value]);

        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(id_array) as ArrayRef,
                Arc::new(content_array) as ArrayRef,
                Arc::new(vector_array) as ArrayRef,
                Arc::new(source_array) as ArrayRef,
            ],
        )
        .context("Failed to create record batch")?;

        let schema_ref = batch.schema();
        let reader = RecordBatchIterator::new(vec![Ok(batch)], schema_ref);
        
        self.table
            .add(reader)
            .execute()
            .await
            .context("Failed to add document to LanceDB")?;

        Ok(())
    }

    async fn search(&self, query_embedding: &[f32]) -> Result<Vec<SearchResult>> {
        let table = self.conn.open_table(self.table.name()).execute().await?;
        let results = table
            .query()
            .limit(self.config.top_k)
            .nearest_to(query_embedding)?
            .execute()
            .await
            .context("Failed to execute LanceDB query")?;

        let batches: Vec<RecordBatch> = results.try_collect().await
            .context("Failed to collect query results")?;

        let mut search_results = Vec::new();
        
        for batch in batches {
            let num_rows = batch.num_rows();
            
            let id_col = batch.column_by_name("id")
                .context("Missing 'id' column")?;
            let content_col = batch.column_by_name("content")
                .context("Missing 'content' column")?;
            let source_col = batch.column_by_name("source")
                .context("Missing 'source' column")?;
            let distance_col = batch.column_by_name("_distance")
                .context("Missing '_distance' column")?;
            
            let id_array = id_col.as_any().downcast_ref::<StringArray>()
                .context("Failed to cast 'id' to StringArray")?;
            let content_array = content_col.as_any().downcast_ref::<StringArray>()
                .context("Failed to cast 'content' to StringArray")?;
            let source_array = source_col.as_any().downcast_ref::<StringArray>()
                .context("Failed to cast 'source' to StringArray")?;
            let distance_array = distance_col.as_any().downcast_ref::<Float32Array>()
                .context("Failed to cast '_distance' to Float32Array")?;
            
            for i in 0..num_rows {
                let id = id_array.value(i).to_string();
                let content = content_array.value(i).to_string();
                let distance = distance_array.value(i);
                
                let mut metadata = std::collections::HashMap::new();
                if !source_col.is_null(i) {
                    metadata.insert("source".to_string(), source_array.value(i).to_string());
                }
                
                let document = Document {
                    id,
                    content,
                    embedding: vec![],
                    metadata,
                };
                
                let score = 1.0 - distance;
                
                search_results.push(SearchResult {
                    document,
                    score,
                });
            }
        }
        
        Ok(search_results)
    }

    async fn count(&self) -> Result<usize> {
        let count = self.table.count_rows(None).await?;
        Ok(count)
    }

    async fn clear(&self) -> Result<()> {
        anyhow::bail!("LanceDB clear not yet fully implemented")
    }

    async fn get_indexed_paths(&self) -> Result<Vec<String>> {
        anyhow::bail!("LanceDB get_indexed_paths not yet fully implemented")
    }

    async fn remove_by_source(&self, _source_path: &str) -> Result<usize> {
        anyhow::bail!("LanceDB remove_by_source not yet fully implemented")
    }
}

impl LanceDbStore {
    fn create_schema(vector_size: u64) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    vector_size as i32,
                ),
                false,
            ),
            Field::new("source", DataType::Utf8, true),
        ]))
    }

    /// Creates a new LanceDB store and ensures the table exists.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory path where LanceDB should store data
    /// * `collection_name` - Name of the table to use
    /// * `vector_size` - Dimension of the embedding vectors
    pub async fn new(config: RagConfig, path: &str, collection_name: &str, vector_size: u64) -> Result<Self> {
        let conn = connect(path)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        let table_names = conn.table_names().execute().await?;
        
        let table = if table_names.contains(&collection_name.to_string()) {
            conn.open_table(collection_name)
                .execute()
                .await
                .context("Failed to open LanceDB table")?
        } else {
            let schema = Self::create_schema(vector_size);

            conn.create_empty_table(collection_name, schema)
                .execute()
                .await
                .context("Failed to create LanceDB table")?
        };

        Ok(Self {
            config,
            conn,
            table,
            vector_size,
        })
    }
}
