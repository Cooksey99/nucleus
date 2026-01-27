//! Test example for HTTP transport connecting to CoinGecko MCP server
//!
//! This example demonstrates how to:
//! 1. Connect to a remote MCP server over HTTP
//! 2. Initialize the MCP connection
//! 3. List available tools
//! 4. Make a sample request (get Bitcoin price)

use nucleus_core::mcp::transport::http::HttpTransport;
use serde_json::json;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("🔗 Testing HTTP connection to CoinGecko MCP Server\n");

    // CoinGecko MCP server endpoints:
    // Public (keyless): https://mcp.api.coingecko.com/mcp
    // Pro (authenticated): https://mcp.pro-api.coingecko.com/mcp
    // Alternative SSE endpoint: https://mcp.api.coingecko.com/sse
    
    let server_url = "https://mcp.api.coingecko.com/mcp";
    println!("📍 Connecting to: {}\n", server_url);

    // Create HTTP transport
    let mut transport = HttpTransport::new(server_url);

    // Step 1: Initialize MCP connection
    // MCP protocol requires an initialize handshake
    println!("📡 Step 1: Initializing MCP connection...");
    
    let init_params = json!({
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {}
        },
        "clientInfo": {
            "name": "nucleus-test-client",
            "version": "0.1.0"
        }
    });

    match transport.request("initialize", Some(init_params)).await {
        Ok(response) => {
            println!("✅ Initialize successful!");
            println!("Response: {}\n", serde_json::to_string_pretty(&response)?);
        }
        Err(e) => {
            println!("❌ Initialize failed: {}", e);
            println!("\n💡 Note: The CoinGecko server might use HTTP streaming or SSE.");
            println!("   Your current HTTP transport uses POST requests.");
            println!("   You may need to implement SSE or HTTP streaming support.\n");
            return Err(e);
        }
    }

    // Step 2: Send initialized notification
    println!("📡 Step 2: Sending initialized notification...");
    match transport.notify("notifications/initialized", None).await {
        Ok(_) => println!("✅ Initialized notification sent\n"),
        Err(e) => {
            println!("⚠️  Warning: Failed to send initialized notification: {}\n", e);
        }
    }

    // Step 3: List available tools
    println!("📡 Step 3: Listing available tools...");
    match transport.request("tools/list", None).await {
        Ok(response) => {
            println!("✅ Tools listed successfully!");
            
            // Show a summary of available tools
            if let Some(tools) = response.get("tools").and_then(|t| t.as_array()) {
                println!("📋 Found {} available tools:", tools.len());
                for (i, tool) in tools.iter().take(5).enumerate() {
                    if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                        println!("   {}. {}", i + 1, name);
                    }
                }
                if tools.len() > 5 {
                    println!("   ... and {} more tools\n", tools.len() - 5);
                } else {
                    println!();
                }
            } else {
                println!("Response: {}\n", serde_json::to_string_pretty(&response)?);
            }
        }
        Err(e) => {
            println!("❌ Failed to list tools: {}\n", e);
        }
    }

    // Step 4: Get Bitcoin price using get_id_coins tool
    println!("📡 Step 4: Getting Bitcoin price and market data...");
    
    let tool_params = json!({
        "name": "get_id_coins",
        "arguments": {
            "id": "bitcoin",
            "market_data": true,
            "localization": false,
            "tickers": false,
            "community_data": false,
            "developer_data": false,
            "sparkline": false
        }
    });

    match transport.request("tools/call", Some(tool_params)).await {
        Ok(response) => {
            println!("✅ Tool call successful!");
            
            // MCP tool responses return data in a "content" array with "text" fields
            if let Some(content) = response.get("content").and_then(|c| c.as_array()) {
                for item in content {
                    if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                        // Parse the JSON text to extract price info
                        if let Ok(coin_data) = serde_json::from_str::<serde_json::Value>(text) {
                            if let Some(name) = coin_data.get("name").and_then(|n| n.as_str()) {
                                println!("📊 Coin: {}", name);
                            }
                            if let Some(symbol) = coin_data.get("symbol").and_then(|s| s.as_str()) {
                                println!("🔤 Symbol: {}", symbol.to_uppercase());
                            }
                            if let Some(market_data) = coin_data.get("market_data") {
                                if let Some(current_price) = market_data.get("current_price") {
                                    if let Some(usd_price) = current_price.get("usd") {
                                        println!("💰 Price: ${:.2}", usd_price);
                                    }
                                }
                                if let Some(price_change_24h) = market_data.get("price_change_percentage_24h") {
                                    if let Some(change) = price_change_24h.as_f64() {
                                        let emoji = if change >= 0.0 { "📈" } else { "📉" };
                                        println!("{} 24h Change: {:.2}%", emoji, change);
                                    }
                                }
                                if let Some(market_cap) = market_data.get("market_cap") {
                                    if let Some(usd_cap) = market_cap.get("usd") {
                                        if let Some(cap) = usd_cap.as_f64() {
                                            println!("💵 Market Cap: ${:.2}", cap);
                                        }
                                    }
                                }
                            }
                            println!();
                        }
                    }
                }
            } else {
                // Fallback: print response structure
                println!("Response structure: {}\n", serde_json::to_string_pretty(&response)?);
            }
        }
        Err(e) => {
            println!("❌ Tool call failed: {}\n", e);
        }
    }

    println!("✨ Test completed!");
    Ok(())
}

