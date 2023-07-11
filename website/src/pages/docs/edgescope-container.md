---
title: AppScope & Cribl Edge in a Container
---

# Using AppScope & Cribl Edge in a Container

**Assumptions**

- You have an Edge Leader running in Cribl.Cloud.
- You want to add a new Edge Node to its Fleet, and that Edge Node will be in a Docker container.
- You want to scope processes running on the host where the new Edge Node’s Docker container resides. 
- You can even scope processes that run on other containers you run on that host.
- If you want to get fancy, you can add more than one Edge Node and then use Scope by Rules to scope processes on the entire Fleet.

### Overview

Here’s what you’ll do in this walkthrough:

1. Set up the new Edge Node.
2. Scope by PID – this instruments one process on one host.
3. Scope by Rule – this can instrument multiple processes on an entire Edge Fleet.

### Setup

In Cribl.Cloud:

1. Click **Manage Edge**.
2. Select the Fleet (`default_fleet` in my case) where you’ll be adding a Linux host.
3. From the **Add/Update Edge Node** drop-down, select **Docker** to open the **Add Docker Node** modal.
4. Click **Copy script** and dismiss the modal.
- Note: Several parameters provided on this modal can alter the contents of the script. Defaults are fine for this example.
- Alternative: Copy the script [from the docs](https://docs.cribl.io/edge/deploy-running-docker).
5. Note the value of **Edge Nodes** at upper left.

On the Linux host we want to observe with Edge:

1. Paste the script into the shell.
2. Add  `-v /var/run/appscope:/var/run/appscope` to the script.
- This step is not necessary if you copied the script from the docs.
3. Execute the script – this will run Edge in a Docker container.   

Return to Cribl.Cloud UI to verify that the new Node is present:

1.  The value of **Edge Nodes** should have increased by 1.
2. Select **List View** (and filtering by host if needed) – the new Edge Node should appear in the list.  

At this point, Edge is running in a Docker container and is connected to the Edge Leader.  

### Scope by PID

Still in Edge’s **List View** tab:

- Click the GUID for the host we’ve just added.
- Click the count of **running processes** at upper left. Now you can observe and interact with a list of processes on this particular Linux Host.

On the Linux host: 

- In the same shell, start a process you want to scope.  For this example, let’s start `top`.  

Back in Edge: 

Within seconds, `top` should appear in the process list. (If you don’t see it, try filtering by Command.) At this point `top` is running but is not yet instrumented by AppScope.

1. Select the `top` command's row to open the **Process: top** drawer.
2. In the **AppScope** tab, select the AppScope **Configuration** we want to use for this process. 
- Start with `A sensible AppScope configuration ...` .  
3. Leave the default **Source **as `in_appscope`.** **
- If you use a different AppScope Source, configure that Source to set **General Settings** > **Optional Settings** > **UNIX socket permissions** to `777`
4. Click **Start monitoring**.  
- Give it a minute 🙂 
- You should see green checkmarks in all the Status columns.

That’s it!  You just scoped your first process!

Now you’ll want to confirm that Edge is receiving AppScope data for the scoped `top` process.

On the Edge Leader:

- Navigate to **More** > **Sources** and select **AppScope** to open the Source page. 
- Select the row where `ID` is `in_appscope` .
- **Enabled** should be toggled to `Yes`.
- **UNIX domain socket** should be toggled to `Yes`.
- Verify that **Unix socket path** is `/var/run/appscope/appscope.sock`.
- Click the **Live Data tab**, and you should now see events flowing!  Yeah!

**Optional: **Try the other two configurations and see how the scoped data changes. For the config that has payloads enabled, `curl` and `wget` are good choices.

### Scope by Rule

On the Linux host: 

- In your shell, start a process you want to scope.  Again, let’s  run `top`.  

On the Edge Leader:

- Navigate to **More** > **Sources** and select **AppScope** to open the Source page. 
- Select the row where `ID` is `in_appscope` .

In the AppScope Rules tab, under Rules, click **Add Rule** and complete the Rule as follows:

- **Process name**: `top`
- **Process argument**: Skip, because Process name and Process argument are mutually exclusive.
- **AppScope config**: Select the  `A sensible AppScope configuration ...` .  

Click **Save**.

Click **Commit**, provide a commit message, and click **Commit and Deploy**.

Wait for the changes to be deployed to the Edge Node – probably around 30 seconds. 

- Click the **Live Data tab**, and you should now see events flowing!  Yeah! … but wait – there’s more …

Back on the Linux host:

- In your shell, start *another* `top` process to prove that any process matching your Rule gets scoped.
- Click the **Live Data tab**, and you should now see events flowing from two `top` processes – they’ll have different PIDs.

This gets to what’s so powerful about Rules: You can start processes *after* setting things up, and they get scoped.

Now return to Edge’s **List View** tab – the **AppScope** column should indicate that both `top` processes are being scoped.

**Optional: **Try the other two configurations and see how the scoped data changes. For the config that has payloads enabled, `curl` and `wget` are good choices.

In real life, this would be the moment to consider creating Routes and Pipelines to send AppScope data to your favorite Destination(s).
