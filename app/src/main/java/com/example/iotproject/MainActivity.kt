package com.example.iotproject

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.iotproject.ui.theme.IOTProjectTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.OutputStreamWriter
import java.net.Socket

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            IOTProjectTheme {
                var currentScreen by remember { mutableStateOf("collector") }
                Scaffold { innerPadding ->
                    when (currentScreen) {
                        "collector" -> SensorDataCollectorScreen(
                            modifier = Modifier.padding(innerPadding),
                            onNavigateToPrediction = { currentScreen = "prediction" }
                        )
                        "prediction" -> PasswordInputScreen(
                            modifier = Modifier.padding(innerPadding),
                            onNavigateBack = { currentScreen = "collector" }
                        )
                    }
                }
            }
        }
    }
}

@Composable
fun SensorDataCollectorScreen(
    modifier: Modifier = Modifier,
    onNavigateToPrediction: () -> Unit
) {
    var filename by remember { mutableStateOf("") }
    var errorMessage by remember { mutableStateOf("") }
    val context = LocalContext.current
    val serviceIntent = remember { Intent(context, SensorService::class.java) }

    val requestPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { isGranted: Boolean ->
            if (isGranted) {
                serviceIntent.putExtra("filename", filename)
                context.startService(serviceIntent)
            }
        }
    )

    val predictionPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
        onResult = { isGranted: Boolean ->
            if (isGranted) {
                onNavigateToPrediction()
            }
        }
    )

    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        OutlinedTextField(
            value = filename,
            onValueChange = {
                filename = it
                errorMessage = ""
            },
            label = { Text("输入文件名") },
            isError = errorMessage.isNotEmpty(),
            supportingText = if (errorMessage.isNotEmpty()) {
                { Text(errorMessage, color = androidx.compose.ui.graphics.Color.Red) }
            } else null,
            modifier = Modifier.padding(bottom = 16.dp)
        )
        Button(
            onClick = {
                // 验证文件名
                when {
                    filename.isBlank() -> {
                        errorMessage = "文件名不能为空"
                        return@Button
                    }
                    !filename.endsWith(".csv") -> {
                        errorMessage = "文件名必须以 .csv 结尾"
                        return@Button
                    }
                }
                errorMessage = ""

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    when (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.POST_NOTIFICATIONS
                    )) {
                        PackageManager.PERMISSION_GRANTED -> {
                            serviceIntent.putExtra("filename", filename)
                            context.startService(serviceIntent)
                        }
                        else -> {
                            requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        }
                    }
                } else {
                    serviceIntent.putExtra("filename", filename)
                    context.startService(serviceIntent)
                }
            },
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            Text("开始收集")
        }
        Button(onClick = { context.stopService(serviceIntent) }) {
            Text("停止收集")
        }
        Button(
            onClick = {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    when (ContextCompat.checkSelfPermission(
                        context,
                        Manifest.permission.POST_NOTIFICATIONS
                    )) {
                        PackageManager.PERMISSION_GRANTED -> {
                            onNavigateToPrediction()
                        }
                        else -> {
                            predictionPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                        }
                    }
                } else {
                    onNavigateToPrediction()
                }
            },
            modifier = Modifier.padding(top = 16.dp)
        ) {
            Text("进入密码预测模式")
        }
    }
}

@Composable
fun PasswordInputScreen(modifier: Modifier = Modifier, onNavigateBack: () -> Unit) {
    var actualPassword by remember { mutableStateOf("") }
    var displayText by remember { mutableStateOf("") }
    val context = LocalContext.current
    val serviceIntent = remember { Intent(context, SensorService::class.java) }
    var isCollecting by remember { mutableStateOf(false) }
    var dataCount by remember { mutableStateOf(0) }
    var statusMessage by remember { mutableStateOf("等待输入密码...") }
    val scope = rememberCoroutineScope()

    // BroadcastReceiver for data count updates
    androidx.compose.runtime.DisposableEffect(Unit) {
        val receiver = object : android.content.BroadcastReceiver() {
            override fun onReceive(ctx: android.content.Context?, intent: android.content.Intent?) {
                dataCount = intent?.getIntExtra("count", 0) ?: 0
            }
        }
        val filter = android.content.IntentFilter("SENSOR_DATA_UPDATE")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(receiver, filter, android.content.Context.RECEIVER_NOT_EXPORTED)
        } else {
            @Suppress("UnspecifiedRegisterReceiverFlag")
            context.registerReceiver(receiver, filter)
        }

        onDispose {
            try {
                context.unregisterReceiver(receiver)
            } catch (e: Exception) {
                // Receiver already unregistered
            }
        }
    }

    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            text = "密码预测训练模式",
            style = androidx.compose.material3.MaterialTheme.typography.headlineSmall,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        Text(
            text = statusMessage,
            style = androidx.compose.material3.MaterialTheme.typography.bodyMedium,
            color = if (isCollecting) androidx.compose.ui.graphics.Color.Green else androidx.compose.ui.graphics.Color.Gray,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = "已收集数据点: $dataCount",
            style = androidx.compose.material3.MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 16.dp)
        )

        OutlinedTextField(
            value = displayText,
            onValueChange = { },
            label = { Text("输入密码（隐藏显示）") },
            readOnly = true,
            visualTransformation = PasswordVisualTransformation(),
            modifier = Modifier.padding(bottom = 16.dp)
        )

        // Number pad layout (0-9)
        androidx.compose.foundation.layout.Column(
            modifier = Modifier.padding(bottom = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Row 1: 1-5
            androidx.compose.foundation.layout.Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier.padding(bottom = 8.dp)
            ) {
                for (i in 1..5) {
                    Button(
                        onClick = {
                            val char = i.toString()
                            actualPassword += char
                            displayText += "*"

                            if (!isCollecting) {
                                val filename = "password_training_${System.currentTimeMillis()}.csv"
                                serviceIntent.putExtra("filename", filename)
                                context.startService(serviceIntent)
                                isCollecting = true
                                statusMessage = "正在收集数据..."

                                // Service启动需要时间，延迟500ms发送label
                                scope.launch {
                                    kotlinx.coroutines.delay(500)
                                    val labelIntent = Intent("UPDATE_LABEL")
                                    labelIntent.setPackage(context.packageName)
                                    labelIntent.putExtra("label", char)
                                    context.sendBroadcast(labelIntent)
                                }
                            } else {
                                // 已经在收集，立即发送
                                val labelIntent = Intent("UPDATE_LABEL")
                                labelIntent.setPackage(context.packageName)
                                labelIntent.putExtra("label", char)
                                context.sendBroadcast(labelIntent)
                            }
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(i.toString())
                    }
                }
            }

            // Row 2: 6-0
            androidx.compose.foundation.layout.Row(
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                for (i in 6..9) {
                    Button(
                        onClick = {
                            val char = i.toString()
                            actualPassword += char
                            displayText += "*"

                            if (!isCollecting) {
                                val filename = "password_training_${System.currentTimeMillis()}.csv"
                                serviceIntent.putExtra("filename", filename)
                                context.startService(serviceIntent)
                                isCollecting = true
                                statusMessage = "正在收集数据..."

                                // Service启动需要时间，延迟500ms发送label
                                scope.launch {
                                    kotlinx.coroutines.delay(500)
                                    val labelIntent = Intent("UPDATE_LABEL")
                                    labelIntent.setPackage(context.packageName)
                                    labelIntent.putExtra("label", char)
                                    context.sendBroadcast(labelIntent)
                                }
                            } else {
                                // 已经在收集，立即发送
                                val labelIntent = Intent("UPDATE_LABEL")
                                labelIntent.setPackage(context.packageName)
                                labelIntent.putExtra("label", char)
                                context.sendBroadcast(labelIntent)
                            }
                        },
                        modifier = Modifier.weight(1f)
                    ) {
                        Text(i.toString())
                    }
                }
                // Add 0 button
                Button(
                    onClick = {
                        val char = "0"
                        actualPassword += char
                        displayText += "*"

                        if (!isCollecting) {
                            val filename = "password_training_${System.currentTimeMillis()}.csv"
                            serviceIntent.putExtra("filename", filename)
                            context.startService(serviceIntent)
                            isCollecting = true
                            statusMessage = "正在收集数据..."

                            // Service启动需要时间，延迟500ms发送label
                            scope.launch {
                                kotlinx.coroutines.delay(500)
                                val labelIntent = Intent("UPDATE_LABEL")
                                labelIntent.setPackage(context.packageName)
                                labelIntent.putExtra("label", char)
                                context.sendBroadcast(labelIntent)
                            }
                        } else {
                            // 已经在收集，立即发送
                            val labelIntent = Intent("UPDATE_LABEL")
                            labelIntent.setPackage(context.packageName)
                            labelIntent.putExtra("label", char)
                            context.sendBroadcast(labelIntent)
                        }
                    },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("0")
                }
            }
        }

        androidx.compose.foundation.layout.Row(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            modifier = Modifier.padding(bottom = 16.dp)
        ) {
            Button(
                onClick = {
                    if (actualPassword.isNotEmpty()) {
                        actualPassword = actualPassword.dropLast(1)
                        displayText = displayText.dropLast(1)
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("删除")
            }

            Button(
                onClick = {
                    actualPassword = ""
                    displayText = ""
                    val labelIntent = Intent("UPDATE_LABEL")
                    labelIntent.setPackage(context.packageName)
                    labelIntent.putExtra("label", "")
                    context.sendBroadcast(labelIntent)
                },
                modifier = Modifier.weight(1f)
            ) {
                Text("清除")
            }
        }

        Button(
            onClick = {
                if (isCollecting) {
                    context.stopService(serviceIntent)
                    isCollecting = false
                    statusMessage = "数据收集完成"

                    // Optionally send to server
                    scope.launch(Dispatchers.IO) {
                        try {
                            val files = context.getExternalFilesDir(null)?.listFiles { file ->
                                file.name.startsWith("password_training_")
                            }
                            files?.forEach { file ->
                                // Here you can send each file to server if needed
                            }
                        } catch (e: Exception) {
                            e.printStackTrace()
                        }
                    }
                }
            },
            enabled = isCollecting,
            modifier = Modifier.padding(bottom = 8.dp)
        ) {
            Text("完成并保存")
        }

        Button(onClick = onNavigateBack) {
            Text("返回")
        }
    }
}

@Preview(showBackground = true)
@Composable
fun SensorDataCollectorScreenPreview() {
    IOTProjectTheme {
        SensorDataCollectorScreen(onNavigateToPrediction = {})
    }
}

@Preview(showBackground = true)
@Composable
fun PasswordInputScreenPreview() {
    IOTProjectTheme {
        PasswordInputScreen(onNavigateBack = {})
    }
}
