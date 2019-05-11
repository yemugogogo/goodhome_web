var today = new Date();

const fileName = "Posen_Points" + today.getMinutes() + today.getSeconds() + today.getMilliseconds() + ".txt";
setTimeout(function () { const value = document.getElementById('myDiv01').value; }, 3000);

document.getElementById('myDiv03').value = value;

//循环执行，每5sec.一次。5sec.后第一次执行。
setInterval(saveTextAsFile(fileName, value), 5000);//5sec.执行一次
//document.getElementById('myDiv02').value = "OK 2";

function saveTextAsFile(_fileName, _value) {
    document.getElementById('myDiv02').value = _fileName;
    //document.getElementById('myDiv03').value = _value;

    var fso = new ActiveXObject(Scripting.FileSystemObject);
    document.getElementById('myDiv02').value = "OK 4";

    var f = fso.createtextfile("D: \"" + _fileName + ".txt", 2, true);
    //document.getElementById('myDiv02').value = "OK 5";

    f.writeLine(_value);
    //document.getElementById('myDiv02').value = "OK 6";

    f.close();
    document.getElementById('myDiv02').value = "OK 7";

}