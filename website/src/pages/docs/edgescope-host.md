---
title: Cribl Edge on a Host
---

# Running AppScope With Cribl Edge on a Host

"Edge Mode" means "driving" AppScope from Cribl Edge. This topic walks you through a setup procedure, and then through the two basic AppScope techniques:

1. **Scope by PID** – Instrumenting one process on one host.
1. **Scope by Rule** – Instrumenting multiple processes on an entire Edge Fleet.

We assume that you have an Edge Leader running in Cribl.Cloud, and that you want to add a new Edge Node to its Fleet, and to scope processes running on the new Edge Node.

You can easily modify these instructions to add more than one Edge Node and then use Scope by Rules to scope processes on the entire Fleet.

## Setting Up a New Edge Node

In Cribl.Cloud:

1. Click **Manage Edge**.
2. Select the Fleet where you’ll be adding a Linux host – `default_fleet` is fine.
3. From the **Add/Update Edge Node** drop-down, select **Linux** > **Add** to open the **Add Linux Node** modal.
4. Click **Copy script** and dismiss the modal.
    - Note: Several parameters provided on this modal can alter the contents of the script. Defaults are fine for this example.
5. Note the value of **Edge Nodes** at upper left.

On the Linux host we want to observe with Edge:

1. Open a shell. 
2. Paste the script into the shell. 
3. Edit the script so that it runs as root, by adding `sudo` to the bash command at the end.
    - For example, `curl 'https://...%2Fcribl' | bash - ` ... should be edited as follows:
    - `curl 'https://... %2Fcribl' | sudo bash - `
1. Execute the script, and give it a minute or so to complete.

Return to Cribl.Cloud UI to verify that the new Node is present:

1.  The value of **Edge Nodes** should have increased by 1.
2. Select **List View** (and filtering by host if needed) – the new Edge Node should appear in the list.  

At this point, Edge is installed on a Linux container in its default location (`/opt/cribl`), is running, and is connected to the Edge Leader.  

## Scoping by PID

Still in Edge’s **List View** tab:

1. Click the GUID for the host we’ve just added.
2. Click the count of **running processes** at upper left to open a list view of processes.

On the Linux host: 

- In a shell, start a process you want to scope.  For this example, let’s start `top`.  

Back in Edge: 

Within seconds, `top` should appear in the process list. (If you don’t see it, try filtering by command.) At this point `top` is running but is not yet instrumented by AppScope.

1. Select the `top` command's row to open the **Process: top** drawer.
2. In the **AppScope** tab, select the AppScope **Configuration** we want to use for this process. 
    - Select `A sensible AppScope configuration ...` .  
3. Leave the default **Source** as `in_appscope`.
    - If you use a different AppScope Source, configure that Source to set **General Settings** > **Optional Settings** > **UNIX socket permissions** to `777`.
4. Click **Start monitoring**.  
    - After a minute or so, you should see green checkmarks in all the Status columns.

Now you’ll want to confirm that Edge is receiving AppScope data for the scoped `top` process.

On the Edge Leader, navigate to **More** > **Sources** and select **AppScope** to open the Source page. 

In the row where `ID` is `in_appscope`: 
- The Source should be enabled.
- Socket should be set to `$CRIBL_HOME/state/appscope.sock`.  
- In the Status column, click **Live**, and you should now see events flowing.

## Scoping by Rule

On the Linux host: 

- In your shell, start a process you want to scope.  Again, let’s  run `top`.  

On the Edge Leader:

- Navigate to **More** > **Sources** and select **AppScope** to open the Source page. 
- Select the row where `ID` is `in_appscope` .

In the AppScope Rules tab, under Rules, click **Add Rule** and complete the Rule as follows:

- **Process name**: `top`
- **Process argument**: Skip this, because **Process name** and **Process argument** are mutually exclusive.
- **AppScope config**: Select `A sensible AppScope configuration ...` .  

Next:

1. Click **Save**.
2. Click **Commit** and provide a commit message. 
3. Click **Commit and Deploy**.

Wait for the changes to be deployed to the Edge Node – probably around 30 seconds. 

- Click the **Live Data tab**, and you should now see events flowing. Next, we'll show that any process matching your Rule gets scoped.

Back on the Linux host:

1. In your shell, start **another** `top` process.
2. Click the **Live Data tab**.
   
You should now see events flowing from two `top` processes – they’ll have different PIDs.

Return to Edge’s **List View** tab – the **AppScope** column should indicate that both `top` processes are being scoped.

This gets to what’s so powerful about Rules: You can start processes **after** setting things up, and they get scoped.

Optionally, you can try the other two configurations and see how the scoped data changes. For the config that has payloads enabled, `curl` and `wget` are good choices.

If you want to do more, consider creating Routes and Pipelines in Cribl Edge, to send AppScope data to your favorite Destination(s).
