// https://github.com/seanmonstar/reqwest/blob/master/examples/simple.rs

#[tokio::main]

async fn main() -> Result<(), reqwest::Error> {
    let res = reqwest::get("https://cribl.io").await?;

    println!("Status: {}", res.status());

    let body = res.text().await?;

    println!("Body:\n\n{}", body);

    Ok(())
}
