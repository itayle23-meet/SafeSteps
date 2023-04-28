package com.example.newas;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;

public class Signed_in extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_signed_in);

        Intent intent = getIntent();
        String value = intent.getStringExtra("key");
    }

    public void help(View view) {
        Intent intent = new Intent(Signed_in.this, Helped.class);
        startActivity(intent);
    }

//    @Override
//    public boolean onCreateOptionsMenu(Menu menu) {
//        getMenuInflater().inflate(R.menu.main_menu, menu);
//        return true;
//    }

//    @Override
//    public boolean onOptionsItemSelected(MenuItem item) {
//        switch (item.getItemId()) {
//            case R.id.explain:
//                Intent intent = new Intent(Signed_in.this, Helped.class);
//                startActivity(intent);
//                return true;
//            case R.id.location:
//                Intent intent2 = new Intent(Signed_in.this, Helped.class);
//                startActivity(intent2);
//                return true;
//            case R.id.links:
//                Intent intent3 = new Intent(Signed_in.this, Helped.class);
//                startActivity(intent3);
//                return true;
//            default:
//                return super.onOptionsItemSelected(item);
//        }
//    }
}