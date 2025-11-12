package com.example.iotproject

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import java.io.File
import java.io.FileOutputStream

class SensorService : Service(), SensorEventListener {

    companion object {
        private const val TAG = "SensorService"
    }

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var rotationVector: Sensor? = null
    private var magnetometer: Sensor? = null

    private var fileOutputStream: FileOutputStream? = null
    private var filename: String? = null

    private var currentLabel: String = ""
    private var dataPointCount = 0

    private val sensorRate = 20000 // Microseconds for 50Hz

    // BroadcastReceiver to handle label updates
    private val labelReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            currentLabel = intent?.getStringExtra("label") ?: ""
            Log.d(TAG, "Label updated to: $currentLabel")
        }
    }

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service onCreate")
        try {
            val intentFilter = IntentFilter("UPDATE_LABEL")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                registerReceiver(labelReceiver, intentFilter, RECEIVER_NOT_EXPORTED)
            } else {
                @Suppress("UnspecifiedRegisterReceiverFlag")
                registerReceiver(labelReceiver, intentFilter)
            }
            Log.d(TAG, "Label receiver registered successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error in onCreate", e)
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "Service onStartCommand")
        try {
            filename = intent?.getStringExtra("filename")
            Log.d(TAG, "Received filename: $filename")

            if (filename.isNullOrEmpty()) {
                Log.e(TAG, "Filename is null or empty, stopping service")
                stopSelf()
                return START_NOT_STICKY
            }

            startForegroundService()

            sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
            rotationVector = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
            magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)

            Log.d(TAG, "Sensors - ACC: ${accelerometer != null}, GYRO: ${gyroscope != null}, " +
                    "ROT: ${rotationVector != null}, MAG: ${magnetometer != null}")

            sensorManager.registerListener(this, accelerometer, sensorRate)
            sensorManager.registerListener(this, gyroscope, sensorRate)
            sensorManager.registerListener(this, rotationVector, sensorRate)
            sensorManager.registerListener(this, magnetometer, sensorRate)

            setupFile()

            Log.d(TAG, "Service started successfully")
            return START_STICKY
        } catch (e: Exception) {
            Log.e(TAG, "Error in onStartCommand", e)
            stopSelf()
            return START_NOT_STICKY
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        Log.d(TAG, "Service onDestroy")

        // 检查 sensorManager 是否已初始化
        try {
            if (::sensorManager.isInitialized) {
                sensorManager.unregisterListener(this)
                Log.d(TAG, "Sensor listeners unregistered")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering sensors", e)
            e.printStackTrace()
        }

        // 分别处理异常，确保都能执行
        try {
            fileOutputStream?.close()
            Log.d(TAG, "File output stream closed")
        } catch (e: Exception) {
            Log.e(TAG, "Error closing file", e)
            e.printStackTrace()
        }

        try {
            unregisterReceiver(labelReceiver)
            Log.d(TAG, "Label receiver unregistered")
        } catch (e: Exception) {
            Log.e(TAG, "Error unregistering receiver", e)
            e.printStackTrace()
        }
    }

    override fun onSensorChanged(event: SensorEvent?) {
        if (event == null || fileOutputStream == null) return

        try {
            val timestamp = System.currentTimeMillis()
            val nanoTime = System.nanoTime()
            val type = when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> "ACC"
                Sensor.TYPE_GYROSCOPE -> "GYRO"
                Sensor.TYPE_ROTATION_VECTOR -> "ROT_VEC"
                Sensor.TYPE_MAGNETIC_FIELD -> "MAG"
                else -> "UNKNOWN"
            }

            val x = event.values.getOrElse(0) { 0f }
            val y = event.values.getOrElse(1) { 0f }
            val z = event.values.getOrElse(2) { 0f }
            val w = event.values.getOrElse(3) { 0f }

            val line = "$timestamp,$nanoTime,$type,$x,$y,$z,$w,$currentLabel\n"

            fileOutputStream?.write(line.toByteArray())
            dataPointCount++

            if (dataPointCount % 50 == 0) {
                val updateIntent = Intent("SENSOR_DATA_UPDATE")
                updateIntent.setPackage(packageName)
                updateIntent.putExtra("count", dataPointCount)
                sendBroadcast(updateIntent)
            }

            // 每1000个数据点输出一次日志
            if (dataPointCount % 1000 == 0) {
                Log.d(TAG, "Data points collected: $dataPointCount")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error in onSensorChanged", e)
            e.printStackTrace()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onBind(intent: Intent?): IBinder? = null

    private fun setupFile() {
        filename?.let {
            try {
                val externalFilesDir = getExternalFilesDir(null)
                Log.d(TAG, "External files dir: $externalFilesDir")

                if (externalFilesDir == null) {
                    Log.e(TAG, "External files directory is null!")
                    stopSelf()
                    return
                }

                // 确保目录存在
                if (!externalFilesDir.exists()) {
                    Log.d(TAG, "Creating external files directory")
                    externalFilesDir.mkdirs()
                }

                val file = File(externalFilesDir, it)
                Log.d(TAG, "Creating file: ${file.absolutePath}")

                fileOutputStream = FileOutputStream(file, true)

                if (file.length() == 0L) {
                    val header = "timestamp,nanoTime,type,x,y,z,w,label\n"
                    fileOutputStream?.write(header.toByteArray())
                    Log.d(TAG, "CSV header written")
                }

                dataPointCount = 0
                Log.d(TAG, "File setup completed successfully")
            } catch (e: Exception) {
                Log.e(TAG, "Error setting up file", e)
                e.printStackTrace()
                stopSelf()
            }
        }
    }

    private fun startForegroundService() {
        val channelId = "sensor_service_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(channelId, "Sensor Data Collection", NotificationManager.IMPORTANCE_LOW)
            getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
        }
        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("IoT Project")
            .setContentText("Collecting sensor data...")
            .setSmallIcon(R.mipmap.ic_launcher)
            .build()

        // Android 14+ 需要指定前台服务类型
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(1, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(1, notification)
        }
        Log.d(TAG, "Foreground service started")
    }
}
